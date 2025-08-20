#!/usr/bin/env python3
"""
LabelImg Launcher - Simple GUI to launch LabelImg
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys

def launch_labelimg():
    """Launch LabelImg application"""
    try:
        # Change to the project directory
        project_dir = "/Users/administrator/Documents/Brand_Detector_AI"
        labelimg_path = "/Users/administrator/Library/Python/3.9/bin/labelImg"
        
        # Launch LabelImg
        subprocess.Popen([labelimg_path], cwd=project_dir)
        messagebox.showinfo("Success", "LabelImg is launching!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch LabelImg: {str(e)}")

def main():
    """Main function"""
    # Create the main window
    root = tk.Tk()
    root.title("LabelImg Launcher")
    root.geometry("300x200")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Create and pack widgets
    title_label = tk.Label(root, text="LabelImg Launcher", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    info_label = tk.Label(root, text="Click the button below to launch LabelImg\nfor annotating Nike and Puma logos", 
                         font=("Arial", 10))
    info_label.pack(pady=10)
    
    launch_button = tk.Button(root, text="Launch LabelImg", command=launch_labelimg,
                             font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
                             relief="raised", padx=20, pady=10)
    launch_button.pack(pady=20)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
