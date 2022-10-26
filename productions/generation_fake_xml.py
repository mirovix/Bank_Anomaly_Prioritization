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
from configs import production_config as pc
from load_data import LoadData


def load_on_db(engine, xml, id_anomaly, sw):
    query = "INSERT INTO " + pc.rx_input_production_name + " (SYSTEM, ID, CONTENUTO) VALUES ("
    new_query = query + "'" + str(sw).replace(' ', '') + "'," + str(id_anomaly) + ",'" + xml + "');"
    try:
        engine.execute(new_query)
    except Exception:
        return


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
    bank_code.text = "060459"
    root.append(bank_code)

    flag_target_invio = et.Element("flagtargetinvio")
    flag_target_invio.text = 'S'
    root.append(flag_target_invio)

    return et.ElementTree(root)


def generation_new_data(engine, max_elements=None):
    target_comp, target_day = LoadData().load_evaluation()
    target = pd.concat([target_comp, target_day])
    target.NDG = target.NDG.astype(str)
    if max_elements is None: max_elements = target.shape[0]
    for i, row in target.loc[np.random.choice(target.index, size=max_elements)].iterrows():
        row.NDG = (pc.len_ndg - len(row.NDG)) * '0' + row.NDG
        load_on_db(engine, et.tostring(generate_xml(row).getroot(), encoding="unicode"),
                   row.ID, row.SOFTWARE)
    print(">> generated %d fake anomalies on RX database\n" % max_elements)