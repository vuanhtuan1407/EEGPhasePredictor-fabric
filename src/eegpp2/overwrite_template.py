from . dataloader import fread_header_labels
from . utils.common_utils import convert_time

def write_file2(predlbstxt, ptmp, pout):
    header, ftmp = fread_header_labels(ptmp)
    fout = open(pout, "w")
    fout.write("%s" % header)
    ic = 0
    SEP_PRED = "\t"
    SEP_TPMP = "\t"
    for predlb in predlbstxt:
        ic += 1
        ltmp = ftmp.readline()
        if ltmp == "":
            break
        if ic == 1:
            if ltmp.__contains__(","):
                SEP_TPMP = ","

        parts_tmp = ltmp.split(SEP_TPMP)
        fout.write("%s%s%s%s%s" % (parts_tmp[0], SEP_TPMP, predlb, SEP_TPMP,
                                   SEP_TPMP.join(parts_tmp[2:])))

    ftmp.close()
    fout.close()


def write_file(ppred, ptmp, pout):
    _, fpred = fread_header_labels(ppred)
    header, ftmp = fread_header_labels(ptmp)
    fout = open(pout, "w")
    fout.write("%s" % header)
    ic = 0
    SEP_PRED = "\t"
    SEP_TPMP = "\t"
    while True:
        ic += 1
        ltmp = ftmp.readline()
        if ltmp == "":
            break
        lpred = fpred.readline()
        if lpred == "":
            break
        if ic == 1:
            if ltmp.__contains__(","):
                SEP_TPMP = ","
            if lpred.__contains__(","):
                SEP_PRED = ","
        parts_tmp = ltmp.split(SEP_TPMP)
        parts_pred = lpred.split(SEP_PRED)
        if ic == 1:
            assert convert_time(parts_pred[2]) == convert_time(parts_tmp[2])
        fout.write("%s%s%s%s%s" % (parts_tmp[0], SEP_TPMP, parts_pred[1], SEP_TPMP,
                                   SEP_TPMP.join(parts_tmp[2:])))

    ftmp.close()
    fpred.close()
    fout.close()


def demo():
    DIR = "/Users/anhnd/CodingSpace/Python/W"
    write_file("%s/raw_K3_EEG3_11h_FULL_LABEL.txt" % DIR,
               "%s/K3_EEG3_11h.txt" % DIR,
               "%s/K3_EEG3_11h_final.txt" % DIR)