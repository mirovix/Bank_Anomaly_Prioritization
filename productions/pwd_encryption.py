#!/usr/bin/env python3

"""
@Author: Miro
@Date: 22/11/2022
@Version: 1.1
@Objective: criptazione delle password per la connessione ai db
@TODO:
"""
import sys

from configs import production_config as pc
from cryptography.fernet import Fernet
from sys import exit
import time
import maskpass


def read_input_passwords():
    print("\n>>start encryption system\n")
    pwd_comp = bytes(maskpass.askpass(prompt="\n>>password dwa discovery comportamenti >> ", mask="*"), 'utf-8')
    pwd_day = bytes(maskpass.askpass(prompt="\n>>password dwa discovery day >> ", mask="*"), 'utf-8')
    pwd_input = bytes(maskpass.askpass(prompt="\n>>password dwa database rx input >> ", mask="*"), 'utf-8')
    pwd_output = bytes(maskpass.askpass(prompt="\n>>password dwa database rx output >> ", mask="*"), 'utf-8')
    return pwd_comp, pwd_day, pwd_input, pwd_output


def write_key(key):
    if pc.testing_flag is True:
        f = open(pc.pwd_key_path, "wb")
        f.write(key)
        f.close()
    else:
        print("\n>> token generated  >> ", str(key, 'utf-8'))
    return key


def write_encrypted_pwd(path, pwd, ref_key):
    f = open(path, "wb")
    f.write(ref_key.encrypt(pwd))
    f.close()


def encryption():
    pwd_comp, pwd_day, pwd_input, pwd_output = read_input_passwords()

    ref_key = Fernet(write_key(Fernet.generate_key()))
    write_encrypted_pwd(pc.pwd_dwa_comp_path, pwd_comp, ref_key)
    write_encrypted_pwd(pc.pwd_dwa_day_path, pwd_day, ref_key)
    write_encrypted_pwd(pc.pwd_rx_input_path, pwd_input, ref_key)
    write_encrypted_pwd(pc.pwd_rx_output_path, pwd_output, ref_key)

    print("\n>> encryption completed\n")
    time.sleep(15)


def read_key():
    ref_key = None
    try:
        if pc.testing_flag is True:
            with open(pc.pwd_key_path) as f:
                ref_key = bytes(''.join(f.readlines()), 'utf-8')
            f.close()
        elif len(sys.argv) > 1:
            ref_key = bytes(str(sys.argv[1]), 'utf-8')
        else: ref_key = bytes(input("\n>> Token for database connections >> "), 'utf-8')
    except Exception as ex:
        print("\n>> token error >> ", ex)
        exit(1)
    return ref_key


def decryption(path_pwd, ref_key):
    pw = None
    try:
        with open(path_pwd) as f:
            pwd_byte = bytes(''.join(f.readlines()), 'utf-8')
        f.close()
    except Exception as ex:
        print("\n>> error password decryption (check the passwords' files) >> ", ex)
        exit(1)

    try:
        pw = str((Fernet(ref_key).decrypt(pwd_byte)), 'utf-8')
    except Exception as ex:
        print("\n>> error decryption (check the token) >> ", ex)
        exit(1)

    return pw


if __name__ == "__main__":
    encryption()
