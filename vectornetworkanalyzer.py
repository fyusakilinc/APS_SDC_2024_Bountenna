import atexit

from enum import Enum

class VectorNetworkAnalyzer_AverageMode(Enum):
    POINT_BY_POINT = 1
    SWEEP_BY_SWEEP = 2

class VectorNetworkAnalyzer:
    def __init__(
        self,

        frequencyRange = ( None, None ),

        frequencyStepRange = ( None, None ),
        frequencyStepDiscrete = None,

        attenuatorRange = ( None, None ),
        attenuatorDiscrete = None,
        preampRange = ( None, None ),
        preampDiscrete = None,

        trackingGeneratorAmplitude = ( None, None )
    ):
        self._usesContext = False
        self._usedConnect = False

        if not isinstance(frequencyRange, tuple) and not isinstance(frequencyRange, list):
            raise ValueError("Frequency range has to be a 2-tuple or a 2-list")
        if len(frequencyRange) != 2:
            raise ValueError("Frequency range has to be a 2-tuple or 2-list")
        if not isinstance(frequencyStepRange, tuple) and not isinstance(frequencyStepRange, list):
            raise ValueError("Frequency step range has to be a 2-tuple or 2-list")
        if len(frequencyStepRange) != 2:
            raise ValueError("Frequency step range has to be a 2-tuple or 2-list")
        if (frequencyStepDiscrete is not None) and not isinstance(frequencyStepDiscrete, tuple) and not isinstance(frequencyStepDiscrete, list):
            raise ValueError("Allowed discrete frequency steps have to be a list or tuple (or None)")
        if not isinstance(attenuatorRange, tuple) and not isinstance(attenuatorRange, list):
            raise ValueError("Attenuator range has to be a 2-list or 2-tuple")
        if len(attenuatorRange) != 2:
            raise ValueError("Attenuator range has to be a 2-list or 2-tuple")
        if (attenuatorDiscrete is not None) and not isinstance(attenuatorDiscrete, tuple) and not isinstance(attenuatorDiscrete, list):
            raise ValueError("Discrete attenuator specification has to be a list or tuple")
        if not isinstance(preampRange, list) and not isinstance(preampRange, tuple):
            raise ValueError("Preamp range has to be a 2-list or 2-tuple")
        if len(preampRange) != 2:
            raise ValueError("Preamp range has to be a 2-list or 2-tuple")
        if (preampDiscrete is not None) and not isinstance(preampDiscrete, list) and not isinstance(preampDiscrete, tuple):
            raise ValueError("Preamp discrete settings have to be a list, a tuple or None")
        if not isinstance(trackingGeneratorAmplitude, list) and not isinstance(trackingGeneratorAmplitude, tuple):
            raise ValueError("Tracking generator amplitude has to be a list or tuple")
        if len(trackingGeneratorAmplitude) != 2:
            raise ValueError("Tracking genreator amplitude has to be a 2-list or 2-tuple")

        self._frequencyRange = frequencyRange
        self._frequencyStepRange = frequencyStepRange
        self._frequencyStepDiscrete = frequencyStepDiscrete
        self._attenuatorRange = attenuatorRange
        self._attenuatorDiscrete = attenuatorDiscrete
        self._preampRange = preampRange
        self._preampDiscrete = preampDiscrete
        self._trackingGeneratorAmplitude = trackingGeneratorAmplitude

    # Overridden methods:

    def _get_id(self):
        raise NotImplementedException("Method not implemented by device")
    def _set_sweep_range(self, start, stop, step = None):
        raise NotImplementedException("Method not implemented by device")
    def _query_trace(self):
        raise NotImplementedException("Method not implemented by device")
    def _set_average(self, naverages, averageMode = VectorNetworkAnalyzer_AverageMode.POINT_BY_POINT):
        raise NotImplementedException("Method not implemented by device")

    # Exposed methods ...

    def id(self):
        return self._get_id()

    def setSweepRange(self, start, stop, step = None):
        pass
    def setAverageMode(self, naverages, averageMode = VectorNetworkAnalyzer_AverageMode.POINT_BY_POINT):
        pass

def queryTrace(self):
        pass
