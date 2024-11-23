"""IceCube I3InferenceModule.

Contains functionality for writing model predictions to i3 files.
"""

from typing import List, Union, Optional, TYPE_CHECKING, Dict, Any
from collections import deque
import time

import numpy as np
from torch_geometric.data import Data, Batch

from graphnet.utilities.config import ModelConfig
from graphnet.deployment import DeploymentModule
from graphnet.data.extractors.icecube import I3FeatureExtractor
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Frame,
        I3Module,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
    )  # pyright: reportMissingImports=false


class I3BatchInferenceModule(I3Module, DeploymentModule):
    """General class for inference on i3 frames."""

    def __init__(
        self,
        context,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model_config: The ModelConfig (or path to it) that summarizes the
                            model used for inference.
            state_dict: Path to state_dict containing the learned weights.
            model_name: The name used for the model. Will help define the
                        named entry in the I3Frame. E.g. "dynedge".
            gcd_file: path to associated gcd file.
            features: the features of the pulsemap that the model is expecting.
            prediction_columns: column names for the predictions of the model.
                               Will help define the named entry in the I3Frame.
                                E.g. ['energy_reco']. Optional.
            pulsemap: the pulsmap that the model is expecting as input.
            load_modules: Additional modules to include for model loading.
            extra_classes: Additional classes to include for model loading.
            device: The computational device to use. Defaults to "cpu".
        """
        I3Module.__init__(
            self, context,
        )
        
        self.AddParameter("pulsemap_extractor","",None)
        self.AddParameter("model_config","",None)
        self.AddParameter("state_dict","",None)
        self.AddParameter("model_name","",None)
        self.AddParameter("gcd_file","",None)
        self.AddParameter("features","",None)
        self.AddParameter("prediction_columns","",None)
        self.AddParameter("pulsemap","",None)
        self.AddParameter("load_modules","",None)
        self.AddParameter("extra_classes","",None)
        self.AddParameter("device","","cpu")
        self.AddParameter("batch_size","", 1)
        
    def Configure(self):
        
        model_config = self.GetParameter("model_config")
        state_dict = self.GetParameter("state_dict")
        prediction_columns = self.GetParameter("prediction_columns")
        load_modules = self.GetParameter("load_modules")
        extra_classes = self.GetParameter("extra_classes")
        device = self.GetParameter("device")
        
        DeploymentModule.__init__(
            self,
            model_config=model_config,
            state_dict=state_dict,
            prediction_columns=prediction_columns,
            load_modules=load_modules,
            extra_classes=extra_classes,
            device=device,
        )

        # Checks
        gcd_file = self.GetParameter("gcd_file")
        assert isinstance(gcd_file, str), "gcd_file must be string"

        # Set Member Variables
        pulsemap_extractor = self.GetParameter("pulsemap_extractor")
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]

        features = self.GetParameter("features")
        if features is None:
            features = self.model._graph_definition._input_feature_names
        self._graph_definition = self.model._graph_definition
        self._pulsemap = self.GetParameter("pulsemap")
        self._gcd_file = gcd_file
        self.model_name = self.GetParameter("model_name")
        self._features = features
        
        self._batch_size = self.GetParameter("batch_size")
        self._frame_buffer = deque()
        self._data_buffer = []
        self._pframe_counter = 0
        
        self._start_time = None
        
        # Set GCD file for pulsemap extractor
        for i3_extractor in self._i3_extractors:
            i3_extractor.set_gcd(i3_file="", gcd_file=self._gcd_file)

    def Process(self):
        if self._start_time is None:
            self._start_time = time.time()
            
        frame = self.PopFrame()
        
        # Put the current frame into the buffer
        self._frame_buffer.append(frame)
        
        if (frame.Stop == I3Frame.Physics):
            # If it is a physics frame, we will process it
            self._pframe_counter += 1
            data = self._create_data_representation(frame=frame)
            self._data_buffer.append(data)
            
        if self._pframe_counter == self._batch_size:
            self._process_buffer()
            # self.info("Batch took {:.2f} seconds".format(time.time() - self._timer)) 
            # self._timer = time.time()
        
    def _process_buffer(self):
        batch = Batch.from_data_list(self._data_buffer).to(self._device)
        self._data_buffer = []  # Reset data buffer
        
        # (batch_size, num_predictions)
        # gpu_time = time.time() 
        predictions = self._apply_model(data=batch)
        # self.info("GPU prediction took {:.2f} seconds".format(time.time() - gpu_time))
              
        # Check dimensions of predictions and prediction columns
        dim = self._check_dimensions(predictions=predictions)
        
        pframe_index = 0
        while self._frame_buffer:
            frame = self._frame_buffer.popleft()
            
            # We push every frame, but only add predictions to P frames
            if (frame.Stop == I3Frame.Physics):
                for i in range(dim):
                    pred = predictions[pframe_index, i]
                    frame.Put(self.model_name + "_" + self.prediction_columns[i], I3Double(float(pred)))
                pframe_index += 1
            self.PushFrame(frame)
        
        self._pframe_counter = 0
        
    def Finish(self):
        if len(self._frame_buffer) > 0:
            self._process_buffer()
        self.info("File took {:.2f} seconds.".format(time.time()-self._start_time))

    
    def _check_dimensions(self, predictions: np.ndarray) -> int:
        if len(predictions.shape) > 1:
            dim = predictions.shape[1]
        else:
            dim = len(predictions)
        try:
            assert dim == len(self.prediction_columns)
        except AssertionError as e:
            self.error(
                f"predictions have shape {dim} but"
                f"prediction columns have [{self.prediction_columns}]"
            )
            raise e

        # assert predictions.shape[0] == 1  # TODO: Delete since batch_size > 1 now! -PW
        return dim

    def _create_dictionary(
        self, dim: int, predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Transform predictions into a dictionary."""
        data = {}
        for i in range(dim):
            try:
                assert len(predictions[:, i]) == 1
                data[self.model_name + "_" + self.prediction_columns[i]] = (
                    I3Double(float(predictions[:, i][0]))
                )
            except IndexError:
                data[self.model_name + "_" + self.prediction_columns[i]] = (
                    I3Double(predictions[0])
                )
        return data

    def _apply_model(self, data: Data) -> np.ndarray:
        """Apply model to `Data` and case-handling."""
        if data is not None:
            predictions = self._inference(data)
            if isinstance(predictions, list):
                if len(predictions) > 1:
                    self.warning(
                        f"{self.__class__.__name__} assumes one Task "
                        f"but got {len(predictions)}. Only the first will"
                        " be used."
                    )
                predictions = predictions[0]
        else:
            self.warning(
                "At least one event has no pulses "
                " - padding {self.prediction_columns} with NaN."
            )
            predictions = np.repeat(
                [np.nan], len(self.prediction_columns)
            ).reshape(-1, len(self.prediction_columns))
        return predictions

    def _create_data_representation(self, frame: I3Frame) -> Data:
        """Process Physics I3Frame into graph."""
        # Extract features
        input_features = self._extract_feature_array_from_frame(frame)
        # Prepare graph data
        if len(input_features) > 0:
            data = self._graph_definition(
                input_features=input_features,
                input_feature_names=self._features,
            )
            return data
        else:
            return None

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        """Apply the I3FeatureExtractors to the I3Frame.

        Arguments:
            frame: Physics I3Frame (PFrame)

        Returns:
            array with pulsemap
        """
        features = None
        for i3extractor in self._i3_extractors:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self._features]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _add_to_frame(self, frame: I3Frame, data: Dict[str, Any]) -> I3Frame:
        """Add every field in data to I3Frame.

        Arguments:
            frame: I3Frame (physics)
            data: Dictionary containing content that will be written to frame.

        Returns:
            frame: Same I3Frame as input, but with the new entries
        """
        assert isinstance(
            data, dict
        ), f"data must be of type dict. Got {type(data)}"
        for key in data.keys():
            if key not in frame:
                frame.Put(key, data[key])
        return frame
