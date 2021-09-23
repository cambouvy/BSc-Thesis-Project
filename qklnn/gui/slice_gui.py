import os
import sys
from collections import OrderedDict

from IPython import embed
import numpy as np
import scipy.stats as stats
import pandas as pd

qlknn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
networks_path = os.path.join(qlknn_root, "networks")
NNDB_path = os.path.join(qlknn_root, "NNDB")
training_path = os.path.join(qlknn_root, "training")
plot_path = os.path.join(qlknn_root, "plots")

sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
sys.path.append(plot_path)

from slicer import prep_df, is_unsafe, process_row
from model import (
    Network,
    NetworkJSON,
    PostprocessSlice,
    ComboNetwork,
    MultiNetwork,
    no_elements_in_list,
    any_element_in_list,
    db,
)

from PyQt4 import QtCore, QtGui

# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure


def prep_moar():
    store = pd.HDFStore("../gen2_7D_nions0_flat.h5")

    slicedim = "Ati"
    style = "mono"

    filter_geq = -120
    filter_less = 120

    itor = None
    frac = 0.05

    nns = OrderedDict()
    dbnns = []
    dbnns.append(ComboNetwork.get_by_id(2096))
    for dbnn in dbnns:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = "_".join([str(el) for el in [dbnn.__class__.__name__, dbnn.id]])
        nns[nn.label] = nn

    df, target_names = prep_df(
        store,
        nns,
        slicedim,
        filter_less=filter_less,
        filter_geq=filter_geq,
        slice=itor,
        frac=frac,
    )

    unsafe = is_unsafe(df, nns, slicedim)


def find_distinct_target_names(no_divsum=True):
    names = set()
    for cls in [Network, ComboNetwork, MultiNetwork]:
        query = cls.select(cls.target_names).distinct()
        if no_divsum:
            query &= no_elements_in_list(
                cls, "target_names", ["div", "plus"], fields=cls.target_names
            )
        names |= set((str(el[0]) for el in query.tuples()))
    return names


class NetworkNameItem(QtGui.QTableWidgetItem):
    def __lt__(self, other):
        return int(self.text()) < int(other.text())


class NetSelectTableWidget(QtGui.QWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.box = QtGui.QHBoxLayout()
        self.setLayout(self.box)
        lst = self.networkList = QtGui.QTableWidget()
        lst.setColumnCount(4)
        lst.setHorizontalHeaderLabels(["a", "b", "c", "d"])
        lst.setSortingEnabled(1)
        query = Network.select()
        lst.setRowCount(query.count())
        for ii, net in enumerate(query):
            lst.setItem(ii, 0, NetworkNameItem(str(net.id)))
        # self.networkList.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.box.addWidget(self.networkList)


class NetFilterWidget(QtGui.QWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.box = QtGui.QHBoxLayout()
        self.setLayout(self.box)

        names = find_distinct_target_names()
        lst = self.networkTypeList = QtGui.QListWidget()
        lst.addItems(["Pure", "Combo", "Multi"])
        lst.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        lst.selectAll()
        size = lst.sizeHintForRow(0) * lst.count() + 2 * lst.frameWidth()
        lst.setMaximumHeight(size)
        self.box.addWidget(self.networkTypeList)
        self.combo = QtGui.QComboBox()
        self.combo.addItems(list(names))
        self.combo.currentIndexChanged.connect(self.selectionchange)

        self.box.addWidget(self.combo)

    def selectionchange(self, ii):
        pass


class NetSelectWidget(QtGui.QWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.box = QtGui.QVBoxLayout()
        self.setLayout(self.box)

        self.netFilter = NetFilterWidget()
        self.box.addWidget(self.netFilter)

        self.netSelectTable = NetSelectTableWidget()
        # self.netSelectTable.box.addStretch()
        self.box.addWidget(self.netSelectTable)


class NetTabWidget(QtGui.QTabWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        netSelector = NetSelectWidget()
        self.addTab(netSelector, "selnet")
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    tabs = NetTabWidget()
    app.setApplicationName("in")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
