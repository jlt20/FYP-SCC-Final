import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import Network_Parameters_new as NP
import System_Parameters_new
import SCC_Lin  
import VDCS_SCC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NetworkParametersApp:
    def __init__(self, root):
        self.root = root
        self.frames_stack = []  # Stack to keep track of frames
        self.current_frame = None  # The currently displayed frame
        self.frame_counter = 1
        # Initialize storage for file contents
        self.file_contents = {key: None for key in ['load', 'gen', 'wind',
                                                    'Gbr', 'Bbr', 'Gsh', 'Bsh', 'num_bus',
                                                    'Vmin', 'Vmax', 'Pijmax', 'S_limit',
                                                    'net_gen', 'net_gen_mat', 'net_load_mat',
                                                    'net_wind_mat', 'linedata_array', 'gen_value',
                                                    'ppc','num_gen','num_lines', 'num_wind', 'Z_LUT'
                                                    ,'Isc', 'x_gen_wind', 'N']}
        self.vis_contents = {key: None for key in ['I_sc_linearised', 'y_lim']}
        self.vis_contents['y_lim'] = 110
        self.pricing_contents = {key: None for key in ['Demand_active', 'Demand_reactive', 'Scenario_tree_active'
                                                       , 'Scenario_tree_reactive', 'NLHR', 'HRS', 'stc']}
        self.setup_ui()
        self.load_config()  # Load the config file at startup
        self.fig, self.ax = plt.subplots()

    def setup_ui(self):
        self.root.title("SCC Linearisation and Pricing")
        self.root.geometry("1100x1100")
        self.root.resizable(True, True)

        self.root.tk.call("source", "azure.tcl")  # Adjust path if necessary
        self.root.tk.call("set_theme", "light")  # Use "light" or "dark" based on preference

        # Initial main frame setup
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.frames_stack.append(main_frame)  # Add the main frame to the frames stack
        self.show_frame(main_frame)  # Show the main frame

        # Configure layout within the main_frame
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_columnconfigure(2, weight=1)

        # Header setup
        header_frame = tk.Frame(main_frame)  # Background color can be added for visibility
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        header_label = tk.Label(header_frame, text="1) Setup Network Configurations", font=("Arial", 16, "bold"))
        header_label.grid(row=0, column=0, padx=10, pady=10)

        # File input widgets setup
        self.net_load_entry = self.create_file_input(main_frame, "net_load.txt", 1)
        self.net_gen_entry = self.create_file_input(main_frame, "net_gen.txt", 2)
        self.net_wind_entry = self.create_file_input(main_frame, "net_wind.txt", 3)
        self.gen_value_entry = self.create_file_input(main_frame, "gen_value.txt", 4)

        # Initialise button setup
        initialise_btn = ttk.Button(main_frame, text="Initialise Network Parameters", command=self.on_initialise, style='Accent.TButton')
        initialise_btn.grid(row=6, column=1, pady=20, sticky="ew")

        # Variable name selection setup
        excluded_keys = ['linedata', 'wind', 'gen', 'load']
        variable_names = [key for key in self.file_contents.keys() if key not in excluded_keys]
        ttk.Label(main_frame, text="Select Variable Name:").grid(row=7, column=0, padx=10, pady=10)
        self.variable_name_combobox = ttk.Combobox(main_frame, values=variable_names, state="readonly")
        self.variable_name_combobox.grid(row=7, column=1, padx=10, pady=10, sticky="ew")
        self.display_var_btn = ttk.Button(main_frame, text="Display Variable Value", command=self.display_variable_value)
        self.display_var_btn.grid(row=7, column=2, padx=10, pady=10)

        page_counter_label = ttk.Label(main_frame, text=f"Page {self.frame_counter}")
        page_counter_label.grid(row=8, column=1, pady=20)

        # Next page button setup
        self.next_btn = ttk.Button(main_frame, text="Next", command=self.go_to_cal_SCC, state="disabled", style='Accent.TButton')
        self.next_btn.grid(row=8, column=2, pady=20, sticky="e")

    def show_frame(self, frame):
        """Display the given frame and hide the current one."""
        if self.current_frame is not None:
            self.current_frame.pack_forget()
        self.current_frame = frame
        self.current_frame.pack(expand=True, fill="both")

    def go_to_cal_SCC(self):
        """Create and switch to a new frame, allowing return to the previous frame."""
        new_frame = ttk.Frame(self.root, padding="10 10 10 10")  # Create a new frame for the next page

        # Configure layout within the new_frame
        new_frame.grid_columnconfigure(0, weight=1)
        new_frame.grid_columnconfigure(1, weight=1)
        new_frame.grid_columnconfigure(2, weight=1)

        # Header setup
        header_frame = tk.Frame(new_frame)  # Background color can be added for visibility
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        header_label = tk.Label(header_frame, text="2) Solve Network for SCC", font=("Arial", 16, "bold"))
        header_label.grid(row=0, column=0, padx=10, pady=10)

        # Add widgets to new_frame
        back_btn = ttk.Button(new_frame, text="Go Back", command=self.go_back) 
        back_btn.grid(row=12, column=0, pady=20, sticky="w")

        # Entry box for Wind Scenarios
        wind_sen_label = tk.Label(new_frame, text="No. of wind_scenario:")
        wind_sen_label.grid(row=1, column=0, pady=5, sticky="e")
        self.wind_sen_var = tk.StringVar()
        wind_sen_entry = tk.Entry(new_frame, textvariable=self.wind_sen_var)
        wind_sen_entry.grid(row=1, column=1, pady=5, sticky="ew")   

        self.Z_LUT_entry = self.create_file_input(new_frame, "Z_LUT.xlsx (Optional)", 2)

        self.Isc_entry = self.create_file_input(new_frame, "Isc.xlsx (Optional)", 3)

        # SCC calculation button setup
        self.start_scc_btn = ttk.Button(new_frame, text="Start VDCS SCC Calculation", command=self.start_VDCS_calculation, style='Accent.TButton')
        self.start_scc_btn.grid(row=4, column=1, pady=20, sticky="ew")

        # Entry box for I_sc_lim
        I_sc_lim_label = tk.Label(new_frame, text="I_sc_lim:")
        I_sc_lim_label.grid(row=5, column=0, pady=5, sticky="e")
        self.I_sc_lim_var = tk.StringVar()
        I_sc_lim_entry = tk.Entry(new_frame, textvariable=self.I_sc_lim_var)
        I_sc_lim_entry.grid(row=5, column=1, pady=5, sticky="ew")   

        # Entry box for m
        m_label = tk.Label(new_frame, text="m (offset):")
        m_label.grid(row=6, column=0, pady=5, sticky="e")
        self.m_var = tk.StringVar()
        m_entry = tk.Entry(new_frame, textvariable=self.m_var)
        m_entry.grid(row=6, column=1, pady=5, sticky="ew") 

        # Combo box for Pricing Method
        pricing_method_label = tk.Label(new_frame, text="Pricing Method:")
        pricing_method_label.grid(row=7, column=0, pady=5, sticky="e")
        self.pricing_method_var = tk.StringVar()
        pricing_method_combo = ttk.Combobox(new_frame, textvariable=self.pricing_method_var)
        pricing_method_combo['values'] = ("Dispatchable", "Restricted / Marginal Unit Price")
        pricing_method_combo.grid(row=7, column=1, pady=5, sticky="ew")

        # SCC linearisation button setup
        self.start_scc_btn = ttk.Button(new_frame, text="Start SCC Linearisation", command=self.start_scc_linearisation, style='Accent.TButton')
        self.start_scc_btn.grid(row=8, column=1, pady=20, sticky="ew")

        # Entry box for node
        node_label = tk.Label(new_frame, text="Node:")
        node_label.grid(row=9, column=0, pady=5, sticky="e")
        self.node_var = tk.StringVar()
        node_entry = tk.Entry(new_frame, textvariable=self.node_var)
        node_entry.grid(row=9, column=1, pady=5, sticky="ew") 

        # Entry box for y_lim
        y_lim_label = tk.Label(new_frame, text="y_lim:")
        y_lim_label.grid(row=10, column=0, pady=5, sticky="e")
        self.y_lim_var = tk.StringVar()
        y_lim_entry = tk.Entry(new_frame, textvariable=self.y_lim_var)
        y_lim_entry.grid(row=10, column=1, pady=5, sticky="ew") 

        # Visualisation button setup
        self.vis_btn = ttk.Button(new_frame, text="Visualise SCC at Node", command=self.start_vis, style='Accent.TButton')
        self.vis_btn.grid(row=12, column=1, pady=20, sticky="ew")

        self.canvas = FigureCanvasTkAgg(self.fig, master=new_frame)
        self.canvas.get_tk_widget().grid(row=11, column=1, padx=10, pady=10)

        self.frame_counter += 1
        page_counter_label = ttk.Label(new_frame, text=f"Page {self.frame_counter}")
        page_counter_label.grid(row=13, column=2, pady=20)

        # Next page button setup
        self.next_btn_2 = ttk.Button(new_frame, text="Next", command=self.go_to_pricing, state="enabled", style='Accent.TButton')
        self.next_btn_2.grid(row=8, column=2, pady=20, sticky="e")

        self.frames_stack.append(new_frame)  # Add the new frame to the stack
        self.show_frame(new_frame)  # Show the new frame

    def start_scc_linearisation(self):
        selected_method = self.pricing_method_var.get()
        I_sc_lim = int(self.I_sc_lim_var.get())
        string = ""
        m = int(self.m_var.get())
        if selected_method == "Dispatchable":
            dispatch = True
            string = "D"
        else:
            dispatch = False
            string = "RM"
        I_sc_linearised, b_opt = SCC_Lin.SCC_linearise(self.file_contents['Isc'], I_sc_lim, m, self.file_contents["x_gen_wind"], self.file_contents['N'], self.file_contents["num_gen"], self.file_contents["num_bus"], dispatchable=dispatch)
        self.vis_contents["I_sc_linearised"] = I_sc_linearised
        VDCS_SCC.export_to_excel(I_sc_linearised, f"I_sc_linearised_{I_sc_lim}_{m}_{string}.xlsx")
        VDCS_SCC.export_to_excel(b_opt, f"b_opt_{I_sc_lim}_{m}_{string}.xlsx")
        self.next_btn_2["state"] = "enabled"

    """
    def go_to_SCC_linearisation(self):
        # Create and switch to a new frame, allowing return to the previous frame.
        new_frame = ttk.Frame(self.root, padding="10 10 10 10")  # Create a new frame for the next page

        # Configure layout within the new_frame
        new_frame.grid_columnconfigure(0, weight=1)
        new_frame.grid_columnconfigure(1, weight=1)
        new_frame.grid_columnconfigure(2, weight=1)

        # Header setup
        header_frame = tk.Frame(new_frame)  # Background color can be added for visibility
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        header_label = tk.Label(header_frame, text="2) Solve Network for SCC", font=("Arial", 16, "bold"))
        header_label.grid(row=0, column=0, padx=10, pady=10)

        # Entry box for Wind Scenarios
        wind_sen_label = tk.Label(new_frame, text="No. of wind_scenario:")
        wind_sen_label.grid(row=1, column=0, pady=5, sticky="e")
        self.wind_sen_var = tk.StringVar()
        wind_sen_entry = tk.Entry(new_frame, textvariable=self.wind_sen_var)
        wind_sen_entry.grid(row=1, column=1, pady=5, sticky="ew")   

        # Entry box for Wind Capacity
        wind_capacity_label = tk.Label(new_frame, text="Max wind_power:")
        wind_capacity_label.grid(row=2, column=0, pady=5, sticky="e")
        self.wind_capacity_var = tk.StringVar()
        wind_capacity_entry = tk.Entry(new_frame, textvariable=self.wind_capacity_var)
        wind_capacity_entry.grid(row=2, column=1, pady=5, sticky="ew")   

        # SCC calculation button setup
        self.start_scc_btn = ttk.Button(new_frame, text="Start SCC Calculation", command=self.start_scc_calculation, style='Accent.TButton')
        self.start_scc_btn.grid(row=3, column=1, pady=20, sticky="ew")

        excluded_keys = ["I_sc_linearised"]
        variable_names = [key for key in self.SCC_contents.keys() if key not in excluded_keys]
        ttk.Label(new_frame, text="Select Variable Name:").grid(row=4, column=0, padx=10, pady=10)
        self.variable_name_combobox = ttk.Combobox(new_frame, values=variable_names, state="readonly")
        self.variable_name_combobox.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
        self.display_var_btn = ttk.Button(new_frame, text="Display Variable Value", command=self.display_variable_value)
        self.display_var_btn.grid(row=4, column=2, padx=10, pady=10)

        # Entry box for I_sc_lim
        I_sc_lim_label = tk.Label(new_frame, text="I_sc_lim:")
        I_sc_lim_label.grid(row=5, column=0, pady=5, sticky="e")
        self.I_sc_lim_var = tk.StringVar()
        I_sc_lim_entry = tk.Entry(new_frame, textvariable=self.I_sc_lim_var)
        I_sc_lim_entry.grid(row=5, column=1, pady=5, sticky="ew")   

        # Entry box for m
        m_label = tk.Label(new_frame, text="m (offset):")
        m_label.grid(row=6, column=0, pady=5, sticky="e")
        self.m_var = tk.StringVar()
        m_entry = tk.Entry(new_frame, textvariable=self.m_var)
        m_entry.grid(row=6, column=1, pady=5, sticky="ew") 

        # SCC linearisation button setup
        self.start_scc_btn = ttk.Button(new_frame, text="Start SCC Linearisation", command=self.start_scc_linearisation, style='Accent.TButton')
        self.start_scc_btn.grid(row=7, column=1, pady=20, sticky="ew")

        # Entry box for node
        node_label = tk.Label(new_frame, text="Node:")
        node_label.grid(row=8, column=0, pady=5, sticky="e")
        self.node_var = tk.StringVar()
        node_entry = tk.Entry(new_frame, textvariable=self.node_var)
        node_entry.grid(row=8, column=1, pady=5, sticky="ew") 

        # Entry box for y_lim
        y_lim_label = tk.Label(new_frame, text="y_lim:")
        y_lim_label.grid(row=9, column=0, pady=5, sticky="e")
        self.y_lim_var = tk.StringVar()
        y_lim_entry = tk.Entry(new_frame, textvariable=self.y_lim_var)
        y_lim_entry.grid(row=9, column=1, pady=5, sticky="ew") 

        # Visualisation button setup
        self.vis_btn = ttk.Button(new_frame, text="Visualise SCC at Node", command=self.start_vis, style='Accent.TButton')
        self.vis_btn.grid(row=10, column=1, pady=20, sticky="ew")

        self.canvas = FigureCanvasTkAgg(self.fig, master=new_frame)
        self.canvas.get_tk_widget().grid(row=11, column=1, padx=10, pady=10)

        # Add widgets to new_frame
        back_btn = ttk.Button(new_frame, text="Go Back", command=self.go_back) 
        back_btn.grid(row=12, column=0, pady=20, sticky="w")

        self.frame_counter += 1
        page_counter_label = ttk.Label(new_frame, text=f"Page {self.frame_counter}")
        page_counter_label.grid(row=12, column=2, pady=20)

        self.frames_stack.append(new_frame)  # Add the new frame to the stack
        self.show_frame(new_frame)  # Show the new frame
    """

    def go_to_pricing(self):
        new_frame = ttk.Frame(self.root, padding="10 10 10 10")

        # Configure layout within the new_frame
        new_frame.grid_columnconfigure(0, weight=1)
        new_frame.grid_columnconfigure(1, weight=1)
        new_frame.grid_columnconfigure(2, weight=1)

        # Header setup
        header_frame = tk.Frame(new_frame)  # Background color can be added for visibility
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        header_label = tk.Label(header_frame, text="3) Pricing Configuration", font=("Arial", 16, "bold"))
        header_label.grid(row=0, column=0, padx=10, pady=10)

        self.Load_Demand_entry = self.create_file_input(new_frame, "Load_Demand.xlsx (Active and Reactive)", 1)

        self.Wind_Scenario_entry = self.create_file_input(new_frame, "Wind_Scenario_Tree.xlsx (Active and Reactive)", 2)

        # Entry box for Stages
        stages_label = tk.Label(new_frame, text="No. of stages:")
        stages_label.grid(row=4, column=0, pady=5, sticky="e")
        self.stages_var = tk.StringVar()
        stages_entry = tk.Entry(new_frame, textvariable=self.stages_var)
        stages_entry.grid(row=4, column=1, pady=5, sticky="ew")   

        # Entry box for Days
        days_label = tk.Label(new_frame, text="No. of days:")
        days_label.grid(row=3, column=0, pady=5, sticky="e")
        self.days_var = tk.StringVar()
        days_entry = tk.Entry(new_frame, textvariable=self.days_var)
        days_entry.grid(row=3, column=1, pady=5, sticky="ew")   

        # self.NLHR_entry = self.create_file_input(new_frame, "NLHR.txt", 4)
        self.NLHR_entry = self.create_file_input(new_frame, "NLHR.txt", 5)

        # self.HRS_entry = self.create_file_input(new_frame, "HRS.txt", 5)
        self.HRS_entry = self.create_file_input(new_frame, "HRS.txt", 6)

        # self.stc_entry = self.create_file_input(new_frame, "stc.txt", 6)
        self.stc_entry = self.create_file_input(new_frame, "stc.txt", 7)

        """
        # Visualisation button setup
        self.setup_sys_btn = ttk.Button(new_frame, text="Set up System Parameters", command=self.setup_sys, style='Accent.TButton')
        self.setup_sys_btn.grid(row=7, column=1, pady=20, sticky="ew")
        """

        # Visualisation button setup
        self.setup_sys_btn = ttk.Button(new_frame, text="Set up System Parameters", command=self.setup_sys, style='Accent.TButton')
        self.setup_sys_btn.grid(row=8, column=1, pady=20, sticky="ew")

        # Add widgets to new_frame
        back_btn = ttk.Button(new_frame, text="Go Back", command=self.go_back) 
        back_btn.grid(row=12, column=0, pady=20, sticky="w")

        self.frame_counter += 1
        page_counter_label = ttk.Label(new_frame, text=f"Page {self.frame_counter}")
        page_counter_label.grid(row=13, column=2, pady=20)

        self.frames_stack.append(new_frame)
        self.show_frame(new_frame)

    def setup_sys(self):
        stages = int(self.stages_var.get())
        days = int(self.days_var.get())
        Demand_active = self.pricing_contents["Demand_active"] * 1e-2
        Demand_reactive = self.pricing_contents["Demand_reactive"] * 1e-2
        num_load = Demand_active.shape[0]

        Wind_scenarios_active = self.pricing_contents["Scenario_tree_active"] * 1e-2
        Wind_scenarios_reactive = self.pricing_contents["Scenario_tree_reactive"] * 1e-2

        NLHR = self.pricing_contents["NLHR"][0]
        HRS = self.pricing_contents["HRS"][0]
        stc = self.pricing_contents["stc"][0]
        num_wind = Wind_scenarios_active.shape[1]
        num_wind_gen = len(Wind_scenarios_active)

        Wind_scenarios_active = np.reshape(Wind_scenarios_active, (days, num_wind_gen, stages))
        Wind_scenarios_reactive = np.reshape(Wind_scenarios_reactive, (days, num_wind_gen, stages))

        num_wind = Wind_scenarios_active.shape[1]

        Demand_active_day_matrix = Demand_active.copy()
        Demand_reactive_day_matrix = Demand_reactive.copy()

        Demand_active_day_cell = np.empty((days, stages), dtype=object)
        Demand_reactive_day_cell = np.empty((days, stages), dtype=object)
        Wind_scenarios_day_active_cell = np.empty((days, stages), dtype=object)
        Wind_scenarios_day_reactive_cell = np.empty((days, stages), dtype=object)

        for j in range(stages):
            Demand_active_day_cell[0, j] = Demand_active_day_matrix[:,j]
            Demand_reactive_day_cell[0, j] = Demand_reactive_day_matrix[:,j]
            Wind_scenarios_day_active_cell[0, j] = Wind_scenarios_active[:,:,j]
            Wind_scenarios_day_reactive_cell[0, j] = Wind_scenarios_reactive[:,:,j]

        Demand_day_active = np.empty((1, 1), dtype=object)
        Demand_day_reactive = np.empty((1, 1), dtype=object)
        Wind_scenarios_day_active = np.empty((1, 1), dtype=object)
        Wind_scenarios_day_reactive = np.empty((1, 1), dtype=object)

        Demand_day_active[0,0] = Demand_active_day_cell
        Demand_day_reactive[0,0] = Demand_reactive_day_cell
        Wind_scenarios_day_active[0,0] = Wind_scenarios_day_active_cell
        Wind_scenarios_day_reactive[0,0] = Wind_scenarios_day_reactive_cell

        scenarios = 1

        num_gen = len(HRS)
        array1 = np.array(self.file_contents['gen_value'][0]) / 100
        array1 = np.reshape(array1, (num_gen,1))

        Gen_limits_active = np.zeros((num_gen, 1))
        Gen_limits_active = np.concatenate([Gen_limits_active, array1], axis=1)

        array2 = np.ones((num_gen,1)) * 100
        Gen_limits_reactive = -np.ones((num_gen, 1)) * 100
        Gen_limits_reactive = np.concatenate([Gen_limits_reactive, array2], axis=1)

        Ramp_limits = np.ones((num_gen,2)) * 30
        Updowntime_limits = np.zeros((num_gen,2))

        if Gen_limits_active.shape[0] != num_gen:
            raise ValueError('Matrices "Gen_limits_active" and "Ramp_limits" must have as many rows as generators defined')
        if Gen_limits_active.shape[1] != 2:
            raise ValueError('Matrices "Gen_limits_active" and "Ramp_limits" must only have 2 columns')
        if not np.array_equal(np.shape(Gen_limits_active), np.shape(Ramp_limits)):
            raise ValueError('There is a problem with the size of matrix "Gen_limits_active" or "Ramp_limits"')
        if not np.array_equal(np.shape(Gen_limits_active), np.shape(Gen_limits_reactive)):
            raise ValueError('There is a problem with the size of matrix "Gen_limits_active" or "Gen_limits_reactive"')

        return days, Demand_active, Demand_day_active, Demand_day_reactive, Demand_reactive, Gen_limits_active, Gen_limits_reactive, HRS, NLHR, num_gen, num_load, num_wind, Ramp_limits, scenarios, stages, stc, Updowntime_limits, Wind_scenarios_active, Wind_scenarios_day_active, Wind_scenarios_day_reactive, Wind_scenarios_reactive
        
    def go_back(self):
        """Return to the previous frame."""
        if len(self.frames_stack) > 1:
            self.frame_counter -= 1
            self.frames_stack.pop()  # Remove the current frame from the stack
            self.show_frame(self.frames_stack[-1])  # Show the previous frame

    def create_file_input(self, parent, file_type, row):
        ttk.Label(parent, text=f"{file_type} File:").grid(row=row, column=0, padx=10, pady=10)
        entry = ttk.Entry(parent)
        entry.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        btn = ttk.Button(parent, text="Browse", command=lambda ft=file_type, e=entry: self.load_file(ft, e))
        btn.grid(row=row, column=2, padx=10, pady=10)
        return entry

    def load_file(self, file_type, entry_field):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("Excel files", "*.xlsx")])
        if file_path:
            entry_field.delete(0, tk.END)
            entry_field.insert(0, file_path)
            file_content = self.read_file_content(file_path)
            if file_content is not None:
                # Save the content based on file type
                if "net_load" in file_type:
                    self.file_contents['load'] = file_content
                    self.save_file_paths("load", file_path)  # Save paths after updating
                elif "net_gen" in file_type:
                    self.file_contents['gen'] = file_content
                    self.save_file_paths("gen", file_path)  # Save paths after updating
                elif "net_wind" in file_type:
                    self.file_contents['wind'] = file_content
                    self.save_file_paths("wind", file_path)  # Save paths after updating
                elif "gen_value" in file_type:
                    self.file_contents['gen_value'] = file_content
                    self.save_file_paths("gen_value", file_path)  # Save paths after updating
                elif "Z_LUT" in file_type:
                    self.file_contents['Z_LUT'] = file_content
                    self.save_file_paths("Z_LUT", file_path)
                elif "Isc" in file_type:
                    self.file_contents['Isc'] = file_content
                    self.save_file_paths("Isc", file_path)
                elif "NLHR" in file_type:
                    self.pricing_contents["NLHR"] = file_content
                elif "HRS" in file_type:
                    self.pricing_contents["HRS"] = file_content
                elif "stc" in file_type:
                    self.pricing_contents['stc'] = file_content

    def read_file_content(self, file_path):
        try:
            if file_path.endswith('.xlsx'):
                # Might want to change this name later
                if 'Demand' in file_path:
                    df1 = pd.read_excel(file_path, sheet_name='Demand_active')
                    df2 = pd.read_excel(file_path, sheet_name='Demand_reactive')
                    # Extract the desired subset of data
                    # subset_active = df1.iloc[1:25, :19]
                    subset_active = df1.iloc[:, :-2]

                    # Find the first row with all NaN values
                    first_empty_row_index = df2.index[df2.isnull().all(axis=1)][0]
                    subset_reactive = df2.iloc[:first_empty_row_index, :-3]

                    Demand_active = subset_active.to_numpy().T
                    Demand_reactive = subset_reactive.to_numpy().T
                    self.pricing_contents['Demand_active'] = Demand_active
                    self.pricing_contents['Demand_reactive'] = Demand_reactive
                # Might want to change this name later
                elif 'wind' in file_path:
                    df3 = pd.read_excel(file_path, sheet_name='Scenario_tree_active', header=None)
                    df4 = pd.read_excel(file_path, sheet_name='Scenario_tree_reactive', header=None)
                    
                    self.pricing_contents['Scenario_tree_active'] = df3.to_numpy().T
                    self.pricing_contents['Scenario_tree_reactive'] = df4.to_numpy().T
                else:
                    df = pd.read_excel(file_path)
                    return df.to_numpy()  # Convert DataFrame to numpy array
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as file:
                    content = []
                    for line in file:
                        cleaned_line = line.strip().strip('[],')
                        components = cleaned_line.split(',')
                        processed_components = []
                        for component in components:
                            try:
                                value = eval(component.strip())
                                processed_components.append(value)
                            except Exception as e:
                                print(f"Error evaluating component: {component}; Error: {e}")
                        content.append(processed_components)
                return content
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {e}")
            return None

    def on_initialise(self):
        try:
            # Assuming NP.network_Param returns values for all keys, adjust as necessary
            (Gbr, Bbr, Gsh, Bsh, num_bus, Vmin, Vmax, Pijmax, S_limit, net_gen, 
            net_gen_mat, net_load_mat, net_wind_mat, linedata_array, ppc) = NP.network_Param(
                self.file_contents['gen'][0], self.file_contents['load'][0], 
                self.file_contents['wind'][0])

            # Update the dictionary with the returned values
            self.file_contents.update({
                'Gbr': Gbr, 
                'Bbr': Bbr, 
                'Gsh': Gsh, 
                'Bsh': Bsh, 
                'num_bus': num_bus, 
                'Vmin': Vmin, 
                'Vmax': Vmax, 
                'Pijmax': Pijmax, 
                'S_limit': S_limit, 
                'net_gen': net_gen, 
                'net_gen_mat': net_gen_mat, 
                'net_load_mat': net_load_mat, 
                'net_wind_mat': net_wind_mat, 
                'linedata_array': linedata_array,
                'ppc': ppc,
                'num_gen':len(net_gen_mat),
                'num_lines':len(linedata_array),
                'num_wind':len(net_wind_mat)
            })

            messagebox.showinfo("Success", "Initialization completed successfully.")
            # Enable the next button after successful initialization
            self.next_btn["state"] = "enabled"
        except Exception as e:
            messagebox.showerror("Error", "Invalid input or file format: " + str(e))

    def display_variable_value(self):      
        var_name = self.variable_name_combobox.get().strip()
        if var_name in self.file_contents:
            value = self.file_contents[var_name]
            messagebox.showinfo("Variable Value", f"Value of {var_name}: {value}")
        elif var_name in self.SCC_contents:
            value = self.SCC_contents[var_name]
            messagebox.showinfo("Variable Value", f"Value of {var_name}: {value}")
        else:
            messagebox.showwarning("Variable Not Found", f"No value found for '{var_name}'. Please make sure the variable name is correct and the network parameters have been initialized.")

    def save_file_paths(self, name, file_path):
        # Read the existing configuration into a dictionary
        config = {}
        try:
            with open('file_paths.cfg', 'r') as config_file:
                for line in config_file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        config[key] = value
        except FileNotFoundError:
            pass  # It's okay if the file doesn't exist yet

        # Update the configuration with the new file path
        config[name] = file_path

        # Write the updated configuration back to the file
        with open('file_paths.cfg', 'w') as config_file:
            for key, value in config.items():
                config_file.write(f"{key}={value}\n")

    def load_config(self):
        # Load file paths from config and update entry fields.
        try:
            with open('file_paths.cfg', 'r') as config_file:
                for line in config_file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        # Depending on the key, update the corresponding entry field
                        if key == 'load':
                            self.net_load_entry.delete(0, tk.END)
                            self.net_load_entry.insert(0, value)
                            file_content = self.read_file_content(value)
                            self.file_contents['load'] = file_content
                        elif key == 'gen':
                            self.net_gen_entry.delete(0, tk.END)
                            self.net_gen_entry.insert(0, value)
                            file_content = self.read_file_content(value)
                            self.file_contents['gen'] = file_content
                        elif key == 'wind':
                            self.net_wind_entry.delete(0, tk.END)
                            self.net_wind_entry.insert(0, value)
                            file_content = self.read_file_content(value)
                            self.file_contents['wind'] = file_content
                        elif key == 'gen_value':
                            self.gen_value_entry.delete(0, tk.END)
                            self.gen_value_entry.insert(0, value)
                            file_content = self.read_file_content(value)
                            self.file_contents['gen_value'] = file_content
        except FileNotFoundError:
            pass  # It's okay if the file doesn't exist yet
    
    def start_VDCS_calculation(self):
        try:
            wind_sen = int(self.wind_sen_var.get())
            N = 2**int(len(self.file_contents['gen'][0]))
            self.file_contents['N'] = N
            Y39, gen_39, complex_voltages, I_L = VDCS_SCC.cal_Y39(self.file_contents['ppc'], self.file_contents['gen'][0], self.file_contents['gen_value'][0], self.file_contents['num_lines'], self.file_contents['linedata_array'], self.file_contents['num_bus'])
            if not self.Z_LUT_entry.get():
                Z_LUT = VDCS_SCC.cal_Z_LUT(self.file_contents['num_gen'], self.file_contents['num_bus'], Y39, gen_39)
                VDCS_SCC.export_to_excel(Z_LUT, "Z_LUT.xlsx")
            else:
                Z_LUT = self.file_contents['Z_LUT']

            if not self.Isc_entry.get():
                Isc, x_gen_wind = VDCS_SCC.cal_VDCS_SCC(wind_sen, self.file_contents['wind'][0], self.file_contents['num_gen'], self.file_contents['num_bus'], complex_voltages, Z_LUT, I_L, self.file_contents['ppc'])
                self.file_contents['Isc'] = Isc
                self.file_contents['x_gen_wind'] = x_gen_wind
            else:
                Isc = self.file_contents['Isc']
                x_gen_wind = np.zeros(((N-1) * wind_sen,1))
                for i_w in range(wind_sen):
                    x_gen_wind[(N-1)*i_w:(N-1)*(i_w+1)] = (i_w + 1)/wind_sen    
                self.file_contents['x_gen_wind'] = x_gen_wind
                # VDCS_SCC.export_to_excel(x_gen_wind, "x_gen_wind.xlsx")

        except Exception as e:
            messagebox.showerror("Error", "Error starting VDCS SCC: " + str(e))
    
    def start_vis(self):
        try:
            if self.y_lim_var.get() == "":
                y_lim = self.vis_contents['y_lim']
            else: 
                y_lim = int(self.y_lim_var.get())
            node_num = int(self.node_var.get())
            I_sc = self.file_contents['Isc']
            I_sc_linearised = self.vis_contents['I_sc_linearised']
            N = self.file_contents['N']
            num_gen = self.file_contents['num_gen']
            wind_sen = int(self.wind_sen_var.get())
            # Plot the graph
            self.plot_graph(I_sc, I_sc_linearised, node_num, N, num_gen, y_lim, wind_sen)

        except Exception as e:
            messagebox.showerror("Error", "Error starting visualisation: " + str(e))

    def plot_graph(self, I_sc, I_sc_linearised, node_num, N, num_gen, y_lim, wind_sen):
        ib = node_num  # Selected node

        # Clear previous plot
        self.ax.clear()

        # Create an array [1:(N-1)*N_wind]'
        x_values = np.arange(1, (N-1)*wind_sen+1)

        # Create an array for labels
        ind1 = np.arange(N-1, (N-1)*wind_sen+1, N-1)
        label = [f'Wind{i}' for i in range(1, wind_sen+1)]

        # Plot the data
        self.ax.plot(x_values, I_sc[:, ib], label='Observed')
        self.ax.plot(x_values, I_sc_linearised[:, ib], label='Linearised')

        # Add vertical lines for each wind scenario
        for i in ind1:
            self.ax.axvline(i, linestyle='--', linewidth=0.7, color='grey')

        # Label the vertical lines
        self.ax.set_xticks(ind1)
        self.ax.set_xticklabels(label)

        # Set legend, limits, and labels
        self.ax.set_xlim([1, (N-1)*wind_sen])
        self.ax.set_ylim([0, y_lim])  # Adjust this limit as per your data
        self.ax.grid(True)
        self.ax.set_xlabel(f'Scenario No. (1~{(N-1)*wind_sen + 1})')
        self.ax.set_ylabel('SCC (p.u.)')
        self.ax.legend()

        # Redraw canvas
        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkParametersApp(root)
    root.mainloop()
