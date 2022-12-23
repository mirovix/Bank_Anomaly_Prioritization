#!/usr/bin/env python3

"""
@Author: Miro
@Date: 04/10/2022
@Version: 1.2
@Objective: lettura del file xml dal database RX
@TODO:
"""

import xml.etree.ElementTree as et
from datetime import datetime
import numpy as np
import pandas as pd
from sys import exit
from configs import production_config as pc, train_config as tc


def find_attributes(root, name_template):
    att_dict = {}
    for item in root.findall('attributes/Attributo'):
        if item.find('Etichetta').text in name_template.keys():
            att_dict.update({item.find('Etichetta').text: str(item.find('Valore').text)})
    return att_dict


def define_xml_prediction(percentuale_value, fascia_value):
    root = et.Element("Gruppo")

    def_desc = et.Element("Etichetta")
    def_desc.text = "Indice Predittivo"
    root.append(def_desc)

    attributes = et.Element("attributes")

    simple = et.SubElement(attributes, "Attributo")
    default_description = et.SubElement(simple, "Etichetta")
    default_description.text = "Priority Rating"
    value = et.SubElement(simple, "Valore")
    value.text = str(percentuale_value)

    simple = et.SubElement(attributes, "Attributo")
    default_description = et.SubElement(simple, "Etichetta")
    default_description.text = "Priority Score"
    value = et.SubElement(simple, "Valore")
    value.text = str(tc.production_soglie[fascia_value])

    root.append(attributes)
    return root


def concatenate_xml_predictions(input_xml_string, score_value, fascia_value):
    main_root = et.fromstring(input_xml_string)
    table_node = [define_xml_prediction(score_value, fascia_value)]

    for main_node in main_root.iter('attributes'):
        for node in main_node.iter('table'):
            table_node.append(node)
            main_node.remove(node)

    for e in main_root.findall('attributes'):
        for node in table_node: e.append(node)

    return et.tostring(et.ElementTree(main_root).getroot())


def from_xml_to_target(input_xml_string, index, comportamenti=False, day=False):
    root, dict_xml, name_template = et.fromstring(input_xml_string), {}, None

    dict_xml.update({pc.xml_names['id']: index})
    dict_xml.update({pc.xml_names['cod_anomalia']: str(root.find('code').text)})
    dict_xml.update({pc.xml_names['sw']: str(root.find('system').text).replace(' ', '')})

    fill_data_len = pc.iso_date_len - len(root.find('createdAt').text)
    data = {pc.xml_names['data_anomalia']: str((root.find('createdAt').text[:-4]) + '0' * fill_data_len)}
    stato = {pc.xml_names['stato']: str('NOT_EVALUATED')}

    if comportamenti is True:
        att_dict = find_attributes(root, pc.dict_target_comp)
        dict_xml.update({pc.xml_names['importo']: str(format(float(att_dict['amount']), '.2f'))})
        dict_xml.update(data)
        dict_xml.update(stato)
        dict_xml.update({pc.xml_names['ndg']: att_dict['NDG']})
    elif day is True:
        att_dict = find_attributes(root, pc.dict_target)
        dict_xml.update({pc.xml_names['importo']: str(format(float(att_dict['amount']), '.2f'))})
        dict_xml.update(data)
        dict_xml.update(stato)
        dict_xml.update({pc.xml_names['ndg']: att_dict['NDG']})
        time_format = datetime.strptime(att_dict['operationDate'], '%Y-%m-%d %H:%M:%S')
        dict_xml.update({pc.xml_names['op_date']: time_format})
        dict_xml.update({pc.xml_names['cod_op']: att_dict['operationCode']})
    else:
        return
    return dict_xml


def write_result(input_predictions, input_rx, query_output_name=pc.rx_output_production_name):
    try:
        input_predictions.replace(np.nan, None, inplace=True)
        for index, row in input_predictions.iterrows():
            input_rx.at[row.ID, pc.xml_col_name] = concatenate_xml_predictions(row.CONTENUTO,
                                                                               row.score,
                                                                               row.fascia)
        input_rx = input_rx.reset_index(level=0)
        input_rx.to_sql(query_output_name, con=pc.engine_rx_output, if_exists='append', index=False)
    except Exception:
        print(">> error: something wrong happened in writing result")


def build_target(sql, engine, max_elements=None):
    sql_commands, xml_template = sql.split(';'), None

    try:
        xml_template = pd.read_sql(sql_commands[0] + pc.name_table_input, engine,
                                   index_col=[pc.index_name]).astype(str)
        if xml_template.shape[0] == 0: return None, None, None

        engine.execute(sql_commands[1] + pc.name_table_input)
        if pc.testing_flag is True: engine.execute(sql_commands[2] + pc.rx_output_production_name)

    except Exception as ex:
        print(">> input data cannot be obtained due to the following error: ", ex)
        exit(1)

    xml_template.replace('NaT', np.nan, inplace=True)
    xml_template.replace('None', np.nan, inplace=True)

    dict_xml_day, dict_xml_comp, i = [], [], 0
    if max_elements is None: max_elements = xml_template.shape[0]

    for index, row in xml_template.head(max_elements).iterrows():
        if row.SYSTEM.replace(' ', '') == pc.system_comp_name:
            dict_xml_comp.append(from_xml_to_target(row.CONTENUTO, index, comportamenti=True))
        elif row.SYSTEM.replace(' ', '') == pc.system_day_name:
            dict_xml_day.append(from_xml_to_target(row.CONTENUTO, index, day=True))
        else:
            continue

    if len(dict_xml_day) + len(dict_xml_comp) < 1:
        write_result(pd.DataFrame(), xml_template.head(max_elements))
        return None, None, None

    return pd.DataFrame(dict_xml_comp), pd.DataFrame(dict_xml_day), xml_template.head(max_elements)
