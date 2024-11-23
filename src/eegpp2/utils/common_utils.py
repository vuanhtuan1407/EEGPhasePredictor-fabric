import platform
from datetime import datetime

import numpy as np
from scipy.stats import norm

FORMAT1 = '%Y.%m.%d.  %H:%M:%S.%f'  # use in original seq files
FORMAT2 = '%Y.%m.%d.  %H:%M:%S'  # use in original label files and use as default format


def convert_time(time_string, offset=946659600000):
    FORMAT1 = '%Y.%m.%d.  %H:%M:%S.%f'
    FORMAT11 = '%Y.%m.%d.  %H:%M:%S'
    if time_string.__contains__("/"):
        FORMAT1 = '%m/%d/%Y  %H:%M:%S.%f'
        FORMAT11 = '%m/%d/%Y  %H:%M:%S.%f'

    try:
        if time_string[-5:].__contains__("."):
            dt_obj = datetime.strptime(time_string,
                                       FORMAT1)
        else:
            dt_obj = datetime.strptime(time_string,
                                       FORMAT11)

        millisec = int(dt_obj.timestamp() * 1000) - offset
    except:
        millisec = -1
    return millisec


def convert_datetime2ms(datetime_str: str, offset=946659600000):
    if datetime_str.__contains__("/"):
        format_seq = '%m/%d/%Y  %H:%M:%S.%f'
        format_lb = '%m/%d/%Y  %H:%M:%S'
    else:
        format_seq = FORMAT1
        format_lb = FORMAT2

    try:
        if datetime_str[-5:].__contains__("."):
            dt_obj = datetime.strptime(datetime_str, format_seq)
        else:
            dt_obj = datetime.strptime(datetime_str, format_lb)

        ms = int(dt_obj.timestamp() * 1000) - offset
    except:
        ms = -1
    return ms


def convert_ms2datetime(ms, offset=946659600000):
    dt = datetime.fromtimestamp((ms + offset) / 1000)
    return str(datetime.strftime(dt, FORMAT2))


def get_os():
    return platform.system()


def get_path_slash():
    os_system = platform.system()
    if os_system == 'Windows':
        return '\\'
    else:
        return '/'


def generate_normal_vector(length, mean=0, std=1.5):
    if length % 2 == 0:
        print("Length of segment weight loss vector must be odd. Automatic increase length by 1.")
        length += 1

    vector = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    pdf_value = norm.pdf(vector, loc=mean, scale=std)
    pdf_value /= pdf_value[length // 2]

    return pdf_value


if __name__ == '__main__':
    print(generate_normal_vector(1))
    # print(convert_ms2datetime(686505744000))
    # print(convert_datetime2ms("2021.10.02. 21:59:52"))
    # print((convert_datetime2ms("2021.10.02. 21:59:52") - convert_datetime2ms("2021.10.02. 10:06:12")) / 4000 + 1)
