import numpy as np
import sounddevice as sd
import soundfile as sf
from math import sin
from math import floor

class Signal():
    def __init__(self, filename, duration, samplerate, frequency):
        if filename == None:
            self.createSignalFromParams(duration, samplerate, frequency)
        else:
            self.createSignalFromFileName(filename)

        self.echo_signal = None
        self.reverse_signal = None
        self.reverb_signal = None
        self.slowDown_signal = None
        self.chorus_signal = None


    def createSignalFromParams(self, duration, samplerate, frequency):
        self.duration = duration
        self.samplerate = samplerate
        self.frequency = frequency
        
        t = np.linspace(0, self.duration, int(self.duration * self.samplerate), endpoint=False, dtype=np.float32)
        self.signal = np.sin(2 * np.pi * self.frequency * t)

        if self.signal.ndim > 1:
            self.signal = self.signal.mean(axis=1)

    def createSignalFromFileName(self, filename):
        signal, samplerate = sf.read(filename, dtype="float32")
        self.signal = signal
        self.samplerate = samplerate
        self.duration = None

        if self.signal.ndim > 1:
            self.signal = self.signal.mean(axis=1)

    def echo_single(self, delay_s = 0.4, decay = 0.4):
        delay_samples = int(self.samplerate * delay_s)
        output = np.zeros(len(self.signal) + delay_samples)

        for i in range(len(self.signal)):
            output[i] = self.signal[i]

        for i in range(delay_samples, len(output)):
            output[i] = output[i] + self.signal[i - delay_samples] * decay

        self.echo_signal = output 

    def reverse(self):
        self.reverse_signal = np.zeros(len(self.signal))
        for i in range(len(self.signal)):
            self.reverse_signal[i] = self.signal[len(self.signal) - 1 - i]

    def slowDown(self, factor:float = 0.5):
        if factor <= 0:
            raise ValueError("Factor must be greater than 0.")
        
        new_length = int(len(self.signal) / factor)
        output = np.zeros(new_length, dtype=np.float32)

        for i in range(new_length):
            original_index = int(i * factor)
            if original_index < len(self.signal):
                output[i] = self.signal[original_index]
            else:
                output[i] = 0.0

        self.slowDown_signal = output

    def allPassFilter(self, input_signal, delay_s:float, gain:float):
        delay_samples = int(self.samplerate * delay_s)
        output = np.zeros_like(input_signal, dtype=np.float32)
    
        for i in range(len(input_signal)):
            x_delayed = input_signal[i - delay_samples] if i >= delay_samples else 0.0
            y_delayed = output[i - delay_samples] if i >= delay_samples else 0.0
            output[i] = -gain * input_signal[i] + x_delayed + gain * y_delayed
    
        return output

    def combFilter(self, input_signal, delay_s:float, gain:float):
        delay_samples = int(self.samplerate * delay_s)
        output = np.copy(input_signal)

        for i in range(delay_samples, len(output)):
            output[i] = output[i] + output[i - delay_samples] * gain

        return output

    def reverb(self, wet:float = 0.3):
        # 1. Přidání 2 sekund ticha na konec, aby dozvuk neskončil s originálem
        tail_samples = int(self.samplerate * 2.0)
        padded_signal = np.pad(self.signal, (0, tail_samples), mode="constant")

        comb_delays_s = [0.0253, 0.0269, 0.0290, 0.0307, 0.0322, 0.0338]
        comb_gain = 0.82

        # 2. Pole musí mít délku prodlouženého signálu
        comb_signal = np.zeros(len(padded_signal))
        for delay in comb_delays_s:
            comb_signal += self.combFilter(padded_signal, delay, comb_gain)
        comb_signal /= len(comb_delays_s)

        diffused = self.allPassFilter(comb_signal, 0.00510, 0.5)
        diffused = self.allPassFilter(diffused, 0.01260, 0.5)

        # 3. Mícháme s prodlouženým zvukem
        output = (1.0 - wet) * padded_signal + wet * diffused
        
        # 4. Normalizace proti ořezu
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val
            
        self.reverb_signal = output.astype(np.float32)

    def chorus(self, base_delay_s=0.02, depth_s=0.003, lfo_freq=1.5, gain=0.7):
        N = len(self.signal)
        t = np.arange(N) / self.samplerate
        delay_s = base_delay_s + depth_s * np.sin(2 * np.pi * lfo_freq * t)
        delay_samples = delay_s * self.samplerate
        dn_floor = np.floor(delay_samples).astype(int)
        dn_up = dn_floor + 1
        dn_diff = delay_samples - dn_floor
        
        idx_floor = np.arange(N) - dn_floor
        idx_up = np.arange(N) - dn_up
        
        valid_floor = idx_floor >= 0
        valid_up = idx_up >= 0
        
        val = np.zeros(N, dtype=np.float32)

        val[valid_floor] += (1 - dn_diff[valid_floor]) * self.signal[idx_floor[valid_floor]]

        val[valid_up] += dn_diff[valid_up] * self.signal[idx_up[valid_up]]
        
        output = self.signal + gain * val
        
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output /= max_val
            
        self.chorus_signal = output.astype(np.float32)
    
    def playSound(self):
        sd.play(self.signal, self.samplerate)
        sd.wait()

    def playEcho(self):
        if self.echo_signal is not None:
            sd.play(self.echo_signal, self.samplerate)
            sd.wait()
        else:
            print("No echo signal available. Please call echo_single() first.")

    def playReverse(self):
        if self.reverse_signal is not None:
            sd.play(self.reverse_signal, self.samplerate)
            sd.wait()
        else:
            print("No reverse signal available. Please call reverse() first.")

    def playReverb(self):
        if self.reverb_signal is not None:
            sd.play(self.reverb_signal, self.samplerate)
            sd.wait()
        else:
            print("No reverb signal available. Please call reverb() first.")

    def playSlowDown(self):
        if self.slowDown_signal is not None:
            sd.play(self.slowDown_signal, self.samplerate)
            sd.wait()
        else:
            print("No slow down signal available. Please call slowDown() first.")

    def playChorus(self):
        if self.chorus_signal is not None:
            sd.play(self.chorus_signal, self.samplerate)
            sd.wait()
        else:
            print("No chorus signal available. Please call chorus() first.")

class SoundEffects():

    def __init__(self):
        pass


if __name__ == "__main__":
    s1 = Signal(None, 2.0, 44100, 600.0)
    s2 = Signal("siren.wav", None, None, None)
    
    s2.playSound()
    s2.chorus()
    s2.playChorus()
