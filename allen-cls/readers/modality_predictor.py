from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('modalityPredictor')
class ModalityPredictor(Predictor):
    """
    Predictor for sequence to sequence models that visualizes attention scores
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
