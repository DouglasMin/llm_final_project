# -*- coding: utf-8 -*-
import glob
from pyhtml2pdf import converter
from PyPDF2 import PdfMerger, PdfReader


def html2pdf(file_path):
    """
    HTML to PDF Convert
    :param file_path: HTML File Path
    :return: True / False
    """
    try:
        file_list = sorted(glob.glob(f'{file_path}/*.html'))
        print(file_list)
        if len(file_list) == 0:
            return False
        for f in file_list:
            converter.convert(f'file://{f}', f'{f.split(".html")[0]}.pdf')
        return True
    except Exception as err:
        return False


def merge_pdf(file_path, file_name):
    """
    Merge PDF File
    :param file_path: PDF File List
    :param file_name: Merge PDF File Name
    :return: True / False
    """
    try:
        file_list = sorted(glob.glob(f'{file_path}/*.pdf'))
        if len(file_list) == 0:
            return False
        merge = PdfMerger()
        for f in file_list:
            merge.append(PdfReader(open(f, 'rb')))
        merge.write(f'{file_path}/{file_name}.pdf')
        return True
    except Exception as err:
         return False


if __name__ == "__main__":
    path = './datasets'   # HTML File Path
    html2pdf(path)
    merge_pdf(path, 'TEST_PDF')     # PDF File Path & Merge PDF File Name