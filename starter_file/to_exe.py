#!/usr/bin/env python3

"""
@Author: Miro
@Date: 21/12/2022
@Version: 1.0
@Objective: script per la generazione dell'eseguibile
@TODO:
"""

import os
import shutil
import win32com.client
from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys

sys.path.extend(['C:\\workspace\\AnomalyPrioritization'])
from extend_modules import extend_modules
extend_modules()

data_to_include = ['C:\\workspace\\AnomalyPrioritization\\queries', 'C:\\workspace\\AnomalyPrioritization\\driver',
                   'C:\\workspace\\AnomalyPrioritization\\model_data']
additional_libs = ['scipy', 'sqlalchemy', 'pyodbc', 'cx_Oracle', 'pymysql', 'productions', 'imblearn', 'pytz']
libs_exclude = ['configs', 'tensorflow']

build_exe_options = {"packages": find_packages() + additional_libs,
                     "include_files": data_to_include,
                     "excludes": libs_exclude}

main_exe = Executable(
    script="C:\\workspace\\AnomalyPrioritization\\productions\\production.py",
    initScript=None,
    targetName="Prioritas.exe",
    icon="C:/Users/mmihailovic/Desktop/general_info/prioritas_logo.ico"
)

pwd_enc_exe = Executable(
    script="C:\\workspace\\AnomalyPrioritization\\productions\\pwd_encryption.py",
    initScript=None,
    targetName="PwdEncryption.exe",
    icon="C:/Users/mmihailovic/Desktop/general_info/pwd.ico"
)

setup(
    name="Prioritas",
    version="1.0",
    author='Miro',
    author_email='miroljubmihailovic98@gmail.com',
    description="Anomaly prioritization @Miro",
    options={"build_exe": build_exe_options},
    executables=[main_exe, pwd_enc_exe],
)

output_dir = "C:/workspace/Prioritas"
how_to_install_dir = "C:/Users/mmihailovic/Desktop/general_info/how_to_install.txt"

dest_install_dir = output_dir + '/how_to_install.txt'
lib_path = output_dir + "/lib"
configs_dir = "/configs"
data_dir = "/data"
pwd_dir = "/password_encrypt"
row_data_dir = "/row_data"

target_test_dir = row_data_dir + "/target_not_processed.csv"
list_values_dir = row_data_dir + "/list_values.csv"
causale_dir = row_data_dir + "/causale_analitica_V2.csv"

sw_dir = "C:/workspace/AnomalyPrioritization"
to_rmv_dir_model = lib_path + "/train/final_model"
to_rmv_dir_plot = lib_path + "/train/plots"
sklearn_libs_path = "/sklearn/.libs/"
ddl_files_path = "/data/ddl_files"

if not os.path.exists(lib_path + configs_dir): os.mkdir(lib_path + configs_dir)
shutil.copytree(sw_dir + configs_dir, lib_path + configs_dir, dirs_exist_ok=True)

if not os.path.exists(output_dir + data_dir): os.mkdir(output_dir + data_dir)
if not os.path.exists(output_dir + data_dir + pwd_dir): os.mkdir(output_dir + data_dir + pwd_dir)
if not os.path.exists(output_dir + data_dir + row_data_dir): os.mkdir(output_dir + data_dir + row_data_dir)
shutil.copyfile(sw_dir + data_dir + target_test_dir, output_dir + data_dir + target_test_dir)
shutil.copyfile(sw_dir + data_dir + list_values_dir, output_dir + data_dir + list_values_dir)
shutil.copyfile(sw_dir + data_dir + causale_dir, output_dir + data_dir + causale_dir)

shutil.copyfile(how_to_install_dir, dest_install_dir)

shutil.copytree(sw_dir + ddl_files_path, lib_path + sklearn_libs_path, dirs_exist_ok=True)


def create_shortcut(original_filepath, shortcut_filepath):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_filepath)
    shortcut.Targetpath = original_filepath
    shortcut.WindowStyle = 7
    shortcut.save()


create_shortcut(lib_path + configs_dir + "/production_config.py", output_dir + "/production_config.lnk")
create_shortcut(lib_path + configs_dir + "/production_db_config.py", output_dir + "/production_db_config.lnk")
