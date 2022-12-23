#!/usr/bin/env python3

"""
@Author: Miro
@Date: 05/10/2022
@Version: 1.0
@Objective: generazione di fake xml anomalie per testing del rx database
@TODO:
"""

import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from sys import exit
from configs import production_config as pc
from input.load_data import LoadData


def load_on_db(engine, df_to_insert):
    try:
        df_to_insert.to_sql(pc.rx_input_production_name, con=engine, if_exists='replace', index=False)
    except Exception as ex:
        print(">> exception loading fake xml: ", ex)
        exit(1)


def generate_xml(data):
    root = et.Element("Messaggio")

    root.append(et.Element("id"))

    attributes = et.Element("attributes")

    for key in pc.dict_target.keys():
        attributo = et.SubElement(attributes, "Attributo")
        etichetta = et.SubElement(attributo, "Etichetta")
        etichetta.text = key
        hilight = et.SubElement(attributo, "Hilight")
        hilight.text = 'false'
        invisible = et.SubElement(attributo, "invisible")
        invisible.text = 'false'
        valore = et.SubElement(attributo, "Valore")
        valore.text = str(data[pc.dict_target[key]])

    table = et.SubElement(attributes, "table")
    et.SubElement(table, "None")

    root.append(attributes)

    system = et.Element("system")
    system.text = data.SOFTWARE
    root.append(system)

    created_at = et.Element("createdAt")
    created_at.text = str(data.DATA) + " UTC"
    root.append(created_at)

    code = et.Element("code")
    code.text = data.CODICE_ANOMALIA
    root.append(code)

    description = et.Element("description")
    description.text = "Ingresso elevato di contante nell ultimo anno"
    root.append(description)

    bank_code = et.Element("bankcode")
    bank_code.text = str(pc.code_bank)
    root.append(bank_code)

    flag_target_invio = et.Element("flagtargetinvio")
    flag_target_invio.text = 'S'
    root.append(flag_target_invio)

    return et.ElementTree(root)


def generation_new_data(engine, max_elements=None):
    target_comp, target_day = LoadData().load_evaluation()
    target = pd.concat([target_comp, target_day])

    target.drop_duplicates(subset=[pc.index_name], keep='last', inplace=True)
    target.NDG = target.NDG.astype(str)

    if max_elements is None: max_elements = target.shape[0]
    list_to_insert, rnd_target, i = [], [], 0
    for _, row in target.loc[np.random.choice(target.index, size=max_elements)].iterrows():
        row.NDG = (pc.len_ndg - len(row.NDG)) * '0' + row.NDG
        list_to_insert.append([row.SOFTWARE, None, row.ID, None, et.tostring(generate_xml(row).getroot(), encoding="unicode"), None])
        rnd_target.append(row.values.tolist())

    load_on_db(engine, pd.DataFrame(list_to_insert, columns=pc.input_rows_name))
    print(">> generated %d fake anomalies on RX database\n" % max_elements)

    return pd.DataFrame(rnd_target, columns=target.columns).astype(target.dtypes).sort_values("ID").reset_index(drop=True)
