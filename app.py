import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import skrf as rf
import tempfile
from PIL import Image, ImageTk
import joblib
import nanovnav2 as nv

# Load the SVM model and scaler
svm_model = joblib.load('s21_svm_model.joblib')
scaler_s21 = joblib.load('scaler_s21.pkl')

start_freq = 2.55e9  # 2.6 GHz
end_freq = 2.75e9  # 2.9 GHz
step_freq = 2e6  # 1 MHz step frequency


class DataCollector:
    def __init__(self, vna, canvas, app, auto_stop_duration=60):
        self.vna = vna
        self.collecting_data = False
        self.auto_stop_duration = auto_stop_duration
        self.start_time = None
        self.position_label = None
        self.data_directory = None
        self.auto_stopped = False  # Flag to indicate auto-stop
        self.app = app  # Reference to the App instance

        if not os.path.exists("data"):
            os.makedirs("data")

        # Initialize the plot
        self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 4]})
        self.fig.subplots_adjust(left=0.1, bottom=0.2)  # Adjust to make space for widgets

        self.line_s11, = self.ax[0].plot([], [], label='S11', color='b')
        self.line_s21, = self.ax[1].plot([], [], label='S21', color='r')

        self.ax[0].set_xlim(start_freq / 1e6, end_freq / 1e6)
        self.ax[0].set_ylim(-30, 5)
        self.ax[0].set_xlabel('Frequency (MHz)')
        self.ax[0].set_ylabel('S11 (dBm)')
        self.ax[0].grid(True)
        self.ax[0].legend()

        self.ax[1].set_xlim(start_freq / 1e6, end_freq / 1e6)
        self.ax[1].set_ylim(-100, -30)
        self.ax[1].set_xlabel('Frequency (MHz)')
        self.ax[1].set_ylabel('S21 (dBm)')
        self.ax[1].grid(True)
        self.ax[1].legend()

        # Embed the plot in the Tkinter canvas
        self.canvas = canvas
        self.plot_widget = FigureCanvasTkAgg(self.fig, master=canvas)
        self.plot_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.data_points = []
        self.ani = FuncAnimation(self.fig, self.update, interval=50)

    def update_plot(self, freqs, s11, s21):
        self.line_s11.set_data(freqs, s11)
        self.line_s21.set_data(freqs, s21)

        for a in self.ax:
            a.relim()
            a.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def submit_label(self, label):
        self.position_label = label
        self.data_directory = f"data/{self.position_label}"
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def start_collecting(self):
        self.collecting_data = True
        self.auto_stopped = False  # Reset the auto-stop flag
        self.data_points = []  # Reset data points for new session
        self.start_time = datetime.datetime.now()
        print(f"Started collecting data for {self.position_label}...")

    def stop_collecting(self):
        if self.collecting_data:
            self.collecting_data = False
            print(f"Stopped collecting data for {self.position_label}...")
            self.save_data(self.position_label)
            if self.auto_stopped:
                self.app.notify_auto_stop()

    def update(self, frame):
        if not self.vna:
            print("VNA not initialized")
            return

        try:
            data = self.vna._query_trace()
            # print("Data received from VNA")
        except Exception as e:
            print(f"Error querying data: {e}")
            return

        if data is None:
            print("No data received from VNA")
            return

        timestamp = datetime.datetime.now().isoformat()

        # Write the raw data to a temporary .s2p file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.s2p') as temp_file:
            temp_file.write(b'! Touchstone file\n')
            temp_file.write(b'# Hz S RI R 50\n')
            for f, s11, s21 in zip(data["freq"], data["s00raw"], data["s01raw"]):
                temp_file.write(f'{f} {s11.real} {s11.imag} {s21.real} {s21.imag} 0 0 0 0\n'.encode())
            temp_file_name = temp_file.name

        raw_network = rf.Network(temp_file_name)
        os.remove(temp_file_name)

        calibrated_network = cal.apply_cal(raw_network)
        calibrated_s11dbm = 20 * np.log10(np.abs(calibrated_network.s[:, 0, 0]))
        calibrated_s21dbm = 20 * np.log10(np.abs(calibrated_network.s[:, 1, 0]))

        s11 = calibrated_s11dbm.tolist()
        s21 = calibrated_s21dbm.tolist()
        frequency = (data["freq"] / 1e6).tolist()

        self.update_plot(frequency, s11, s21)

        if self.collecting_data and not self.auto_stopped:
            self.data_points.append({
                "timestamp": timestamp,
                "s11": s11,
                "s21": s21,
                "frequency": frequency,
            })

            if self.start_time and (datetime.datetime.now() - self.start_time).seconds >= self.auto_stop_duration:
                self.auto_stopped = True
                self.stop_collecting()

    def save_data(self, position_label):
        start_time = datetime.datetime.now()
        file_path = os.path.join(self.data_directory,
                                 f"yusa_{position_label}_{start_time.strftime('%Y%m%d%H%M%S')}.npz")
        np.savez(file_path, **{"data": self.data_points})
        print(f"Data saved to {file_path}")
        self.data_points = []


class App:
    def __init__(self, root, vna, cal):
        self.root = root
        self.root.title("VNA Data Collector and Real-Time Tester")
        self.root.geometry("1920x1080")

        self.vna = vna
        self.cal = cal  # Store cal object

        # Map integers to string labels
        self.label_dict_rev = {
            0: 'right_leg_forward',
            1: 'left_leg_forward',
            2: 'side_by_side'
        }
        self.images = self.preload_images()

        # Load the SVM model and scaler
        self.model = svm_model
        self.scaler = scaler_s21  # Ensure scaler is available as an instance variable

        self.setup_gui()
        self.collector = DataCollector(self.vna, self.canvas, self)  # Pass the App instance

        self.model_running = False
        self.current_label = None  # Initialize current_label here

    def setup_gui(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.label_frame = ttk.LabelFrame(frame, text="Data Collection")
        self.label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.data_label = ttk.Label(self.label_frame, text="Position Label:")
        self.data_label.grid(row=0, column=0, padx=5, pady=5)

        self.position_label = tk.StringVar()
        self.position_combo = ttk.Combobox(self.label_frame, textvariable=self.position_label)
        self.position_combo['values'] = ('right_leg_forward', 'left_leg_forward', 'side_by_side')
        self.position_combo.grid(row=0, column=1, padx=5, pady=5)
        self.position_combo.bind("<<ComboboxSelected>>", self.update_label)

        self.start_button = ttk.Button(self.label_frame, text="Start", command=self.start_collecting, state=tk.DISABLED)
        self.start_button.grid(row=0, column=2, padx=5, pady=5)

        self.stop_button = ttk.Button(self.label_frame, text="Stop", command=self.stop_collecting, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        # Create a frame for model testing buttons
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.test_button = ttk.Button(model_frame, text="Load and Test Model", command=self.start_model_testing)
        self.test_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_test_button = ttk.Button(model_frame, text="Stop Model", command=self.stop_model_testing)
        self.stop_test_button.grid(row=0, column=1, padx=10, pady=10)
        self.stop_test_button.config(state=tk.DISABLED)

        self.prediction_frame = ttk.LabelFrame(frame, text="Prediction Result")
        self.prediction_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Make the label font larger
        self.prediction_label = ttk.Label(self.prediction_frame, text="Predicted Position: ", font=("Arial", 16))
        self.prediction_label.grid(row=0, column=0, padx=10, pady=10)

        # Create a canvas for the plot
        self.canvas = ttk.Frame(frame)
        self.canvas.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.image_frame = ttk.Frame(frame)
        self.image_frame.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")

        # Resize the image label to be bigger
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.start_button.config(state=tk.NORMAL)

    def update_label(self, event):
        self.collector.submit_label(self.position_label.get())
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def start_collecting(self):
        self.collector.start_collecting()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_collecting(self):
        self.collector.stop_collecting()
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

    def notify_auto_stop(self):
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

    def start_model_testing(self):
        self.model_testing_running = True
        self.run_model_testing()
        self.test_button.config(state=tk.DISABLED)
        self.stop_test_button.config(state=tk.NORMAL)

    def stop_model_testing(self):
        self.model_testing_running = False
        self.test_button.config(state=tk.NORMAL)
        self.stop_test_button.config(state=tk.DISABLED)
        self.prediction_label.config(text="")  # Clear the predicted label
        self.image_label.config(image='')  # Clear the image
        self.current_label = None  # Reset current label

    def run_model_testing(self):
        if self.model_testing_running:
            test_data = self.vna._query_trace()

            def predict_and_update():
                s21_data = np.log10(np.abs(test_data["s01raw"])) * 20

                # Prepare the data
                s21_data = self.scaler.transform([s21_data])  # Use the instance variable scaler

                # Predict position using the SVM model
                predicted_class = self.model.predict(s21_data)[0]

                # Ensure the label dictionary correctly maps the predicted classes
                label_dict = {
                    'right_leg_forward' : 0,
                    'left_leg_forward' : 1,
                    'side_by_side' : 2
                }

                # Check if the predicted class is in the label dictionary
                if predicted_class in label_dict:
                    predicted_label = label_dict[predicted_class]

                    self.prediction_label.config(text=f"Predicted Position: {predicted_class}")

                    if predicted_class != self.current_label:
                        self.display_image(predicted_label)  # Pass the integer class directly
                        self.current_label = predicted_class
                else:
                    print(f"Predicted class {predicted_class} not found in label dictionary")

            threading.Thread(target=predict_and_update).start()

            # Re-run the method after a delay to continuously update the prediction
            self.root.after(100, self.run_model_testing)  # Adjust the interval as needed

    def preload_images(self):
        images = {}
        for idx, label in self.label_dict_rev.items():
            image_path = f"images/{idx}.jpg"
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image = image.resize((400, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                images[idx] = photo
            else:
                print(f"Image {image_path} not found.")
        return images

    def display_image(self, predicted_label):
        try:
            image = self.images[predicted_label]
            self.image_label.config(image=image)
            self.image_label.image = image
        except KeyError:
            print(f"No image found for label {predicted_label}")


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


if __name__ == "__main__":
    calibration_file_path = 'cal_final.cal'  # replace with the actual path
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

    with nv.NanoVNAV2("COM7", debug=False) as vna:
        vna._set_sweep_range(start_freq, end_freq, step_freq)
        root = tk.Tk()
        app = App(root, vna, cal)
        root.mainloop()
