#!/usr/bin/env python3

"""
@Author: Miro
@Date: 24/11/2022
@Version: 1.0
@Objective: processare il target per eliminare errori di valutazione tramite commenti degli operatori
@TODO:
"""

import json
import sys
from googletrans import Translator
from unidecode import unidecode
import keyboard
from load_data import LoadData
import re
import time
import os
from sys import exit


load_data = LoadData()
translator = Translator()
output_file = "output/target.csv"
iter_file = "iter_processed/save_iter.json"


def change_target_value(target, comment, error=False):
    if error is True: to_mod = 'NOT_TO_ALERT'
    else: to_mod = 'TO_ALERT'

    target.loc[target.BODY == comment, 'STATO'] = to_mod
    sys.stdout.write('>> VALUE MODIFIED' + "\r")
    time.sleep(0.10)


def save_iter(i):
    with open(iter_file, 'w') as fp:
        json.dump(i, fp)


def print_instructions():
    print("\n>> System for processing target information\n")
    print(">> keyboard commands\n")
    print(" >> skip >> right arrow")
    print(" >> back >> left arrow")
    print(" >> modify target (TO ALERT) >> ctrl")
    print(" >> correct anomaly (NOT TO ALERT) >> alt")
    print(" >> exit >> esc and save\n\n")


def process_old_iter():
    i = 0
    if os.path.isfile(iter_file):
        response = input(">> iter file detected, continue the past pre-processing (Y/N): ")
        if response.lower() == 'y':
            with open(iter_file, 'rb') as fp:
                i = int(json.load(fp))
                load_data.evaluation_csv = output_file
        print("\n\n\n")
    return i


def df_to_process(target):
    positive_ndg = list(dict.fromkeys(target[target.STATO == 'TO_ALERT'].NDG))
    condition = (target.NDG.isin(positive_ndg)) & (target.STATO == 'NOT_TO_ALERT')
    target_to_examine = target[condition].drop_duplicates('BODY')
    target_to_examine.sort_values(['NDG', 'DATA'], inplace=True)
    target_to_examine.reset_index(drop=True, inplace=True)
    return target_to_examine


def comment_processing(row, i, i_max):
    testo = unidecode(re.sub(' +', ' ', str(row.BODY)).replace('\n', ' ').replace('\r', ' '))
    comment = translator.translate(unidecode(testo), dest='it').text
    to_write = ">> " + str(i + 1) + "/" + str(i_max + 1) + " >> NDG " + \
               str(row.NDG) + " >> DATA ANOMALIA " + str(row.DATA) + " >> STATO " + \
               str(row.STATO) + " >> " + comment
    sys.stdout.write(to_write + "\r")


def save(target, i):
    target.to_csv(output_file)
    save_iter(i)


if __name__ == "__main__":
    print_instructions()

    i = process_old_iter()
    target_ori = load_data.load_evaluation_not_processed(index_col=False, comment_col=False, file_name=False)
    target = load_data.load_evaluation_not_processed(index_col=False, comment_col=True, file_name=False)
    target_to_examine = df_to_process(target)

    while i < target_to_examine.shape[0]:
        os.system('cls')
        flag_slip = False
        row = target_to_examine.iloc[i]

        if i % 20 == 0: save(target_ori, i)

        comment_processing(row, i, target_to_examine.shape[0])
        while True:
            keyboard.read_key()
            if keyboard.is_pressed('esc'):
                save(target_ori, i)
                sys.stdout.write('>> process completed' + "\r")
                exit(0)
            elif keyboard.is_pressed('ctrl'):
                change_target_value(target, row.BODY)
                break
            elif keyboard.is_pressed('alt'):
                change_target_value(target, row.BODY, error=True)
                break
            elif keyboard.is_pressed('right'):
                break
            elif keyboard.is_pressed('left'):
                if i != 0: i -= 1
                flag_slip = True
                break
        if flag_slip is False: i += 1
    save(target_ori, i)
    sys.stdout.write('>> process completed' + "\r")
    exit(0)
