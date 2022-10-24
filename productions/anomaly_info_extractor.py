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
from configs import production_config as pc, train_config as tc


def find_attributes(root, name_template):
    att_dict = {}
    for item in root.findall('attributes/Attributo'):
        if item.find('Etichetta').text in name_template.keys():
            att_dict.update({item.find('Etichetta').text: str(item.find('Valore').text)})
    return att_dict


def concatenate_xml_predictions(input_xml_string, percentuale_value, fascia_value):
    root = et.Element("group")

    id_group = et.Element("id")
    id_group.text = "Indice Predittivo"
    root.append(id_group)

    def_desc = et.Element("defaultDescription")
    def_desc.text = "Indice Predittivo"
    root.append(def_desc)

    attributes = et.Element("attributes")

    simple = et.SubElement(attributes, "simple")
    defaultDescription = et.SubElement(simple, "defaultDescription")
    defaultDescription.text = "Percentuale indice predittivo"
    value = et.SubElement(simple, "value")
    value.text = str(percentuale_value)

    simple = et.SubElement(attributes, "simple")
    defaultDescription = et.SubElement(simple, "defaultDescription")
    defaultDescription.text = "Fascia indice predittivo"
    value = et.SubElement(simple, "value")
    value.text = tc.soglie[fascia_value]

    root.append(attributes)

    return input_xml_string + et.tostring(et.ElementTree(root).getroot(), encoding="unicode")


def from_xml_to_target(input_xml_string, index, comportamenti=False, day=False):
    root, dict_xml, name_template = et.fromstring(input_xml_string), {}, None

    dict_xml.update({pc.xml_names['id']: index})
    dict_xml.update({pc.xml_names['cod_anomalia']: str(root.find('code').text)})
    dict_xml.update({pc.xml_names['sw']: str(root.find('system').text)})

    fill_data_len = pc.iso_date_len - len(root.find('createdAt').text)
    data = {pc.xml_names['data_anomalia']: str((root.find('createdAt').text[:-4]) + '0' * fill_data_len)}
    stato = {pc.xml_names['stato']: str('NOT_EVALUATED')}

    if comportamenti is True:
        amount_table, amount = root.findall('attributes/table/row'), 0.0
        for item in amount_table: amount += float(item[1].text)
        att_dict = find_attributes(root, pc.dict_target_comp)
        dict_xml.update({pc.xml_names['importo']: str(format(amount, '.2f'))})
        dict_xml.update(data)
        dict_xml.update(stato)
        dict_xml.update({pc.xml_names['ndg']: att_dict['NDG']})
    elif day is True:
        att_dict = find_attributes(root, pc.dict_target)
        dict_xml.update({pc.xml_names['importo']: att_dict['amount']})
        dict_xml.update(data)
        dict_xml.update(stato)
        dict_xml.update({pc.xml_names['ndg']: att_dict['NDG']})
        time_format = datetime.strptime(att_dict['operationDate'], '%Y-%m-%d %H:%M:%S')
        dict_xml.update({pc.xml_names['op_date']: time_format})
        dict_xml.update({pc.xml_names['cod_op']: att_dict['operationCode']})
    else:
        return
    return dict_xml


def build_target(sql, engine, max_elements=None):
    sql_commands = sql.split(';')
    xml_template = pd.read_sql(sql_commands[0] + pc.name_table_input, engine,
                               index_col=[pc.index_name]).astype(str)
    if xml_template.shape[0] == 0: return None, None

    engine.execute(sql_commands[1] + pc.name_table_input)
    if pc.testing_flag is True: engine.execute(sql_commands[2] + pc.rx_output_production_name)

    xml_template.replace('NaT', np.nan, inplace=True)
    xml_template.replace('None', np.nan, inplace=True)

    dict_xml, i, single_dict_xml = [], 0, None
    if max_elements is None: max_elements = xml_template.shape[0]
    for index, row in xml_template.iterrows():
        if i > max_elements: break
        i += 1
        if row.SYSTEM.replace(' ', '') == pc.system_comp_name:
            single_dict_xml = from_xml_to_target(row.CONTENUTO, index, comportamenti=True)
        elif row.SYSTEM.replace(' ', '') == pc.system_day_name:
            single_dict_xml = from_xml_to_target(row.CONTENUTO, index, day=True)
        else:
            continue
        dict_xml.append(single_dict_xml)
    return pd.DataFrame(dict_xml), xml_template.head(max_elements)
