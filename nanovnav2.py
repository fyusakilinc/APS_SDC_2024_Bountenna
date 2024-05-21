import sys

sys.path.append("../")

import serial
import atexit
import struct
import math
import time
import numpy as np

import logging

from time import sleep
import vectornetworkanalyzer

from labdevices.exceptions import CommunicationError_ProtocolViolation
from labdevices.exceptions import CommunicationError_Timeout
from labdevices.exceptions import CommunicationError_NotConnected

import skrf as rf
from skrf.calibration import OnePort


# Spectrum analyzer wrapper class
#
# This uses the NanoVNA V2 only on port2 (since the tracking generator
# is not disable-able). It's assumed that port1 is termianted with 50 Ohms

class NanoVNAV2SpectrumAnalyzerPort2:
    def __init__(
            self,
            port,

            logger=None,
            debug=False,
            loglevel=logging.ERROR
    ):
        self._vna = NanoVNAV2(port, logger, debug, loglevel)


class NanoVNAV2(vectornetworkanalyzer.VectorNetworkAnalyzer):
    def __init__(
            self,
            port,

            logger=None,
            debug=False,
            loglevel=logging.ERROR
    ):
        super().__init__(
            frequencyRange=(50e3, 6000e6),
            frequencyStepRange=(1e3, 10e6),
            attenuatorRange=(0, 0),
            preampRange=(0, 0),
            trackingGeneratorAmplitude=(-7, -7)
        )

        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.addHandler(logging.StreamHandler(sys.stderr))
            self._logger.setLevel(loglevel)

        self._debug = debug
        self._discard_first_point = True

        self._regs = {
            0x00: {'mnemonic': "sweepStartHz", 'regbytes': 8, 'fifobytes': None, 'desc': "Sweep start frequency in Hz",
                   "enable": True},
            0x10: {'mnemonic': "sweepStepHz", 'regbytes': 8, 'fifobytes': None, 'desc': "Sweep step frequency in Hz",
                   "enable": True},
            0x20: {'mnemonic': "sweepPoints", 'regbytes': 2, 'fifobytes': None,
                   'desc': "Number of sweep frequency points", "enable": True},
            0x22: {'mnemonic': "valuesPerFrequency", 'regbytes': 2, 'fifobytes': None,
                   'desc': "Number of values to sample and output per data point", "enable": True},
            0x26: {'mnemonic': "sampleMode", 'regbytes': 1, 'fifobytes': None,
                   'desc': "0 is VNA data, 1 is raw data and 2 exits USB data mode", "enable": False},
            0x30: {'mnemonic': "valuesFIFO", 'regbytes': None, 'fifobytes': 32,
                   'desc': "VNA sweep data points. Writing anything clears the FIFO", "enable": True},
            0x40: {'mnemonic': "averageSetting", 'regbytes': 1, 'fifobytes': None,
                   'desc': "Number of samples to average", "enable": True},
            0x41: {'mnemonic': "si5351power", 'regbytes': 1, 'fifobytes': None, 'desc': "SI5351 power",
                   "enable": False},
            0x42: {'mnemonic': "adf4350power", 'regbytes': 1, 'fifobytes': None, 'desc': "ADF4350 power",
                   "enable": True},
            0xEE: {'mnemonic': "lcddump", 'regbytes': 1, 'fifobytes': None, 'desc': "Dump LCD data", "enable": False},
            0xF0: {'mnemonic': "deviceVariant", 'regbytes': 1, 'fifobytes': None,
                   'desc': "The device type (0x02 for the NanoVNA v2)", "enable": True},
            0xF1: {'mnemonic': "protocolVersion", 'regbytes': 1, 'fifobytes': None,
                   'desc': "The protocol version (0x01)", "enable": True},
            0xF2: {'mnemonic': "hardwareRevision", 'regbytes': 1, 'fifobytes': None, 'desc': "Hardware revision",
                   "enable": True},
            0xF3: {'mnemonic': "firmwareMajor", 'regbytes': 1, 'fifobytes': None, 'desc': "Major firmware version",
                   "enable": True},
            0xF4: {'mnemonic': "firmwareMinor", 'regbytes': 1, 'fifobytes': None, 'desc': "Minor firmware version",
                   "enable": True}
        }

        self._sweepStartHz = None
        self._sweepStepHz = None
        self._sweepPoints = None
        self._sweepPoints = None
        self._valuesPerFrequency = None
        self._deviceVariant = None
        self._protocolVersion = None
        self._hardwareRevision = None
        self._firmwareVersion = (None, None)
        self._frequencies = None

        if isinstance(port, serial.Serial):
            self._port = port
            self._portName = None
            self._initialRequests()
        else:
            self._portName = port
            self._port = None

        atexit.register(self.__close)

    # Context management
    def __enter__(self):
        if self._usedConnect:
            raise ValueError("Cannot use context management (with) on a connected port")

        if (self._port is None) and (not (self._portName is None)):
            self._port = serial.Serial(
                self._portName,
                baudrate=2e6,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=15
            )
            self._initialRequests()
        self._usesContext = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close()
        self._usesContext = False

    def __close(self):
        atexit.unregister(self.__close)
        if (not (self._port is None)) and (not (self._portName is None)):
            # Leave USB mode
            try:
                self._reg_write(0x26, 2)
            except:
                # Ignore any error while leaving USB mode
                pass

            self._port.close()
            self._port = None

    # Connect and disconnect

    def _connect(self):
        if (self._port is None) and (not (self._portName is None)):
            self._port = serial.Serial(
                self._portName,
                baudrate=2e6,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=15
            )
            self._initialRequests()
        return True

    def _disconnect(self):
        if not (self._port is None):
            self.__close()
        return True

    # Register access

    def _reg_read(self, address):
        if address not in self._regs:
            raise ValueError(f"Address {address} not supported in NanoVNA library")
        if not self._regs[address]['enable']:
            raise ValueError(f"Access to {self._reg[address]['mnemonic']} not enabled in NanoVNA library")

        # Do read ...
        if self._regs[address]['regbytes'] is None:
            raise ValueError(f"Access to {self._reg[address]['mnemonic']} not possible as register")
        elif self._regs[address]['regbytes'] == 1:
            self._port.write(struct.pack('<BB', 0x10, address))
            value = struct.unpack('<B', self._port.read(1))[0]
        elif self._regs[address]['regbytes'] == 2:
            self._port.write(struct.pack('<BB', 0x11, address))
            value = struct.unpack('<H', self._port.read(2))[0]
        elif self._regs[address]['regbytes'] == 4:
            self._port.write(struct.pack('<BB', 0x12, address))
            value = struct.unpack('<I', self._port.read(4))[0]
        elif self._regs[address]['regbytes'] == 8:
            self._port.write(struct.pack('<BB', 0x12, address))
            valueLow = struct.unpack('<I', self._port.read(4))[0]
            self._port.write(struct.pack('<BB', 0x12, address + 4))
            valueHigh = struct.unpack('<I', self._port.read(4))[0]
            value = valueHigh * 4294967296 + valueLow
        else:
            raise ValueError(
                f"Access width {self._regs[address]['regbytes']} not supported for {self._regs[address]['mnemonic']}")

        return value

    def _reg_write(self, address, value):

        # Do write ...
        if self._regs[address]['regbytes'] == 1:
            self._port.write(struct.pack('<BBB', 0x20, address, value))
        elif self._regs[address]['regbytes'] == 2:
            self._port.write(struct.pack('<BBH', 0x21, address, value))
        elif self._regs[address]['regbytes'] == 4:
            self._port.write(struct.pack('<BBI', 0x22, address, value))
        elif self._regs[address]['regbytes'] == 8:
            self._port.write(struct.pack('<BBQ', 0x23, address, value))
        else:
            raise ValueError(
                f"Access width {self._regs[address]['regbytes']} not supported for {self._regs[address]['mnemonic']}")

    def _op_indicate(self):
        if self._port is None:
            raise CommunicationError_NotConnected("Device is not connected")

        self._port.write(struct.pack('<B', 0x0D))
        resp = struct.unpack('<B', self._port.read(1))[0]

        return resp

    def _op_nop(self):
        if self._port is None:
            raise CommunicationError_NotConnected("Device is not connected")

        self._port.write(struct.pack('<B', 0x00))

    def _initialRequests(self):
        # Send a few no-operation bytes to terminate any lingering
        # commands ...
        for i in range(64):
            self._op_nop()

        # Now read in a loop until we are not able to read any more data
        while True:
            dta = self._port.read(1)
            if not dta:
                break

        # Check if indicate really returned ASCII 2 ...
        indicateResult = self._op_indicate()
        if indicateResult != 0x32:
            raise CommunicationError_ProtocolViolation(
                f"Would expect device to report version 2 (0x32, 50), received {indicateResult}")

        # Write initial values
        #   Initial frequency: 500 MHz (applying first point discard)
        #   4 MHz steps
        #   101 samples (100 + discarded first point)
        #   1 value per frequency
        self._reg_write(0x00, int(500e6) - 4)
        self._reg_write(0x10, int(35))
        self._reg_write(0x20, 101)
        self._reg_write(0x22, 1)

        if self._discard_first_point:
            self._frequencies = np.linspace(500e6 - 35, 500e6 + 101 * 35, 101 + 1)
        else:
            self._frequencies = np.linspace(500e6, 500e6 + 101 * 35, 101)

        # Load current state from registers ...
        self._sweepStartHz = self._reg_read(0x00)
        self._sweepStepHz = self._reg_read(0x10)
        self._sweepPoints = self._reg_read(0x20)
        self._valuesPerFrequency = self._reg_read(0x22)
        self._deviceVariant = self._reg_read(0xF0)
        self._protocolVersion = self._reg_read(0xF1)
        self._hardwareRevision = self._reg_read(0xF2)
        self._firmwareVersion = (self._reg_read(0xF3), self._reg_read(0xF4))

    # Overriden methods from base class
    def _set_sweep_range(self, start, stop, step=50e3):
        # Calculate number of segments, round up to next integer
        # and calculate new end of sweep

        wndPoints = 101
        frqStart = start
        if self._discard_first_point:
            # We have to modify start one point lower and have to account for
            # overlapping windows ... so only 100 points instead of 101 per window since
            # we include the previous last one
            start = start - step
            wndPoints = 100

        nPointsTotal = int(int((stop - start) / int(step)))
        nSegments = int(nPointsTotal / wndPoints)
        if nSegments == 0:
            nSegments = 1
        else:
            nSegments = math.ceil(nSegments)

        # Might differ ...
        # nPointsTotal = nSegments * wndPoints

        stop = start + step * wndPoints * nSegments

        self._sweepStartHz = start
        self._sweepStepHz = step
        self._sweepPoints = nPointsTotal
        self._sweepSegments = nSegments
        self._valuesPerFrequency = 1

        # Create frequencies array ...
        self._frequencies = np.linspace(frqStart, frqStart + nPointsTotal * step, (nPointsTotal) + 1)

        return True

    def __complex_divide(self, a, b):
        return (
            a[0] * b[0] + a[1] * b[1] / (b[0] * b[0] + b[1] * b[1]),
            a[1] * b[0] + a[0] * b[1] / (b[0] * b[0] + b[1] * b[1])
        )

    def _query_trace(self):
        if self._port is None:
            raise CommunicationError_NotConnected("Device it not connected, failed to query data")

        currentStart = self._sweepStartHz
        currentValuesPerFreq = self._valuesPerFrequency
        freqBaseIndex = 0

        pkgdata = {
            "freq": np.asarray([]),
            "fwd0": np.asarray([]),
            "rev0": np.asarray([]),
            "rev1": np.asarray([]),
            "s00raw": np.asarray([]),
            "s01raw": np.asarray([])
        }
        # Now iterate over each segment ...
        for iSegment in range(self._sweepSegments):

            # Set parameters for sweep
            realSweepPoints = self._sweepPoints
            if self._discard_first_point:
                realSweepPoints = realSweepPoints + 1

            self._reg_write(0x00, int(currentStart))
            self._reg_write(0x10, int(self._sweepStepHz))
            self._reg_write(0x20, realSweepPoints)
            self._reg_write(0x22, int(self._valuesPerFrequency))

            # Clear FIFO ...
            self._port.write(struct.pack('<BBB', 0x20, 0x30, 0x00))

            # Read data ...
            nPointsToRead = self._valuesPerFrequency * realSweepPoints
            nDataPoints = nPointsToRead
            while nPointsToRead > 0:
                batchPoints = min(nPointsToRead, 255)
                self._port.write(struct.pack('<BBB', 0x18, 0x30, batchPoints))
                nBytesToRead = 32 * batchPoints
                nBytesRead = 0
                alldata = None
                while (alldata is None) or (nBytesRead < nBytesToRead):
                    datanew = self._port.read(nBytesToRead - nBytesRead)
                    if datanew is None:
                        raise CommunicationError_Timeout("Failed to receive FIFO data")
                    if alldata is not None:
                        alldata = alldata + datanew
                    else:
                        alldata = datanew

                    nBytesRead = nBytesRead + len(datanew)

                nPointsToRead = nPointsToRead - batchPoints

            # If we have to discard the first data point - drop ...
            if self._discard_first_point:
                nDataPoints = nDataPoints - 1
                alldata = alldata[32:]

            # Decode data points of _this_ packet

            newpkgdata = {
                "freq": np.full((nDataPoints), np.nan),
                "fwd0": np.full((nDataPoints), np.nan, dtype=complex),
                "rev0": np.full((nDataPoints), np.nan, dtype=complex),
                "rev1": np.full((nDataPoints), np.nan, dtype=complex),
                "s00raw": np.full((nDataPoints), np.nan, dtype=complex),
                "s01raw": np.full((nDataPoints), np.nan, dtype=complex)
            }

            for iPoint in range(nDataPoints):
                packet = alldata[iPoint * 32: (iPoint + 1) * 32]
                fwd0Re, fwd0Im, rev0Re, rev0Im, rev1Re, rev1Im, freqIndex, _, _ = struct.unpack('<iiiiiiHHI', packet)

                if self._discard_first_point:
                    freqIndex = freqIndex - 1

                newpkgdata["freq"][freqIndex] = self._frequencies[freqIndex + freqBaseIndex]

                newpkgdata["fwd0"][freqIndex] = fwd0Re + 1j * fwd0Im
                newpkgdata["rev0"][freqIndex] = rev0Re + 1j * rev0Im
                newpkgdata["rev1"][freqIndex] = rev1Re + 1j * rev1Im

                # Recover uncalibrated raw data for S parameters by taking into account
                # transmitted signal amplitude and phase

                newpkgdata["s00raw"][freqIndex] = newpkgdata["rev0"][freqIndex] / newpkgdata["fwd0"][freqIndex]
                newpkgdata["s01raw"][freqIndex] = newpkgdata["rev1"][freqIndex] / newpkgdata["fwd0"][freqIndex]

            # Merge to global packet data ...

            for fld in ["freq", "fwd0", "rev0", "rev1", "s00raw", "s01raw"]:
                pkgdata[fld] = np.concatenate((pkgdata[fld], newpkgdata[fld]))

            # Update for next sweep window

            freqBaseIndex = freqBaseIndex + len(newpkgdata["freq"])
            currentStart = currentStart + self._sweepStepHz * 101
            if self._discard_first_point:
                currentStart = currentStart - self._sweepStepHz

        pkgdata["s00rawdbm"] = np.log10(np.absolute(pkgdata["s00raw"])) * 20
        pkgdata["s01rawdbm"] = np.log10(np.absolute(pkgdata["s01raw"])) * 20
        pkgdata["s00rawphase"] = np.angle(pkgdata["s00raw"])
        pkgdata["s01rawphase"] = np.angle(pkgdata["s01raw"])

        return pkgdata

def read_calibration_file(file_path):
    frequencies = []
    short_s = []
    open_s = []
    load_s = []
    thru_s = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            data = list(map(lambda x: float(x.replace(',', '.')), line.split()))
            freq = data[0]
            short_r, short_i = data[1], data[2]
            open_r, open_i = data[3], data[4]
            load_r, load_i = data[5], data[6]
            thru_r, thru_i = data[7], data[8]

            frequencies.append(freq)
            short_s.append([[complex(short_r, short_i), 0], [0, complex(short_r, short_i)]])
            open_s.append([[complex(open_r, open_i), 0], [0, complex(open_r, open_i)]])
            load_s.append([[complex(load_r, load_i), 0], [0, complex(load_r, load_i)]])
            thru_s.append([[0, complex(thru_r, thru_i)], [complex(thru_r, thru_i), 0]])

    freq = rf.Frequency.from_f(frequencies, unit='Hz')

    short_network = rf.Network(frequency=freq, s=np.array(short_s))
    open_network = rf.Network(frequency=freq, s=np.array(open_s))
    load_network = rf.Network(frequency=freq, s=np.array(load_s))
    thru_network = rf.Network(frequency=freq, s=np.array(thru_s))

    return [short_network, open_network, load_network, thru_network]

import tempfile
import os
# Function to animate the plot
def animate(i, vna, lines, ax):
    data = vna._query_trace()

    freqs = data["freq"] / 1e6  # Convert frequency to MHz for plotting
    s11dbm = 20 * np.log10(np.abs(data["s00raw"]))
    s21dbm = 20 * np.log10(np.abs(data["s01raw"]))

    # Write the raw data to a temporary .s2p file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.s2p') as temp_file:
        temp_file.write(b'! Touchstone file\n')
        temp_file.write(b'# Hz S RI R 50\n')
        for f, s11, s21 in zip(data["freq"], data["s00raw"], data["s01raw"]):
            temp_file.write(f'{f} {s11.real} {s11.imag} {s21.real} {s21.imag} 0 0 0 0\n'.encode())
        temp_file_name = temp_file.name

    # Create the Network object from the temporary file
    raw_network = rf.Network(temp_file_name)

    # Delete the temporary file
    os.remove(temp_file_name)

    # Apply calibration to the raw S11 and S21 data
    calibrated_network = cal.apply_cal(raw_network)
    calibrated_s11dbm = 20 * np.log10(np.abs(calibrated_network.s[:, 0, 0]))
    calibrated_s21dbm = 20 * np.log10(np.abs(calibrated_network.s[:, 1, 0]))

    lines.set_data(freqs, calibrated_s11dbm)

    ax.relim()
    ax.autoscale_view()

if __name__ == "__main__":
    from nanovnav2 import NanoVNAV2  # Assuming you have a module `nanovnacal` containing NanoVNAV2

    start_freq = 2.55e9  # 2 GHz
    end_freq = 2.75e9  # 3 GHz
    step_freq = 2e6  # 10 MHz step frequency

    calibration_file_path = 'not_used/cal_255to275.cal'  # replace with the actual path
    my_measured = read_calibration_file(calibration_file_path)

    # Create ideal networks
    freq = my_measured[0].frequency  # Use the same frequency range as the measured data
    short_ideal = rf.Network(frequency=freq, s=np.array([[[-1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]]] * len(freq)))
    open_ideal = rf.Network(frequency=freq, s=np.array([[[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]] * len(freq)))
    load_ideal = rf.Network(frequency=freq, s=np.array([[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]]] * len(freq)))
    thru_ideal = rf.Network(frequency=freq, s=np.array([[[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]]] * len(freq)))

    my_ideals = [short_ideal, open_ideal, load_ideal, thru_ideal]

    # Create a SOLT calibration instance
    cal = rf.SOLT(
        ideals=my_ideals,
        measured=my_measured
    )

    # Run the calibration algorithm
    cal.run()

    with NanoVNAV2("COM7", debug=False) as vna:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib;

        vna._set_sweep_range(start_freq, end_freq, step_freq)

        # Setup the plot for real-time data visualization
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label="Calibrated S11")

        ax.set_ylabel('S11 [dB]')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()

        # FuncAnimation to update the plot continuously
        ani = FuncAnimation(fig, animate, fargs=(vna, line, ax), interval=0)

        plt.show()