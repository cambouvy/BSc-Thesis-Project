import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux, split_parts


class QuaLiKizCommitteeNN(QuaLiKizNDNN):
    """
    Constructs a committee NN from a set of individual NNs with identical
    input and output names. The output of this class is the mean and
    standard deviation of the set of outputs from the individual NNs.

    :arg nns: list. Set of QuaLiKizNDNN which make up the committee NN.
    """

    def __init__(self, nns, low_bound=None, high_bound=None):
        self._nns = list(nns) if isinstance(nns, (list, tuple, np.ndarray)) else None
        self._low_bound = float(low_bound) if isinstance(low_bound, (float, int)) else None
        self._high_bound = float(high_bound) if isinstance(high_bound, (float, int)) else None

        if len(self._nns) > 0:
            for nn in self._nns:
                if np.any(nn._feature_names.ne(self._feature_names)):
                    raise Exception("Supplied NNs have different feature names")
                if np.any(nn._target_names.ne(self._base_target_names)):
                    raise Exception("Supplied NNs have different target names")
            if np.any(self._feature_min > self._feature_max):
                raise Exception("Feature min > feature max")

            if not self._target_names.index.is_unique:
                raise Exception("Non unique index for target_names!")
            # if len(self._target_names) > 1:
            #    raise NotImplementedError('Multiple targets')

    def get_output(
        self,
        input,
        output_pandas=True,
        clip_low=False,
        clip_high=False,
        low_bound=None,
        high_bound=None,
        safe=True,
        **kwargs,
    ):
        if not safe:
            raise Exception("Only safe mode is supported by %s" % (type(self).__name__))
        # standardize to QuaLiKizNDNN, where low_bound and high_bound kwargs are only offered for get_output
        if low_bound is not None:
            self._low_bound = low_bound
        if high_bound is not None:
            self._high_bound = high_bound

        output_eb = kwargs.pop("output_eb") if "output_eb" in kwargs else True
        outlist = [
            np.atleast_2d(
                nn.get_output(
                    input,
                    output_pandas=False,
                    clip_low=False,
                    clip_high=False,
                    low_bound=None,
                    high_bound=None,
                    safe=safe,
                    **kwargs,
                )
            )
            for nn in self._nns
        ]
        outputs = np.dstack(outlist)
        output = np.average(outputs, axis=2)
        if clip_low and self._low_bound is not None:
            output = np.clip(output, self._low_bound, None)
        if clip_high and self._high_bound is not None:
            output = np.clip(output, None, self._high_bound)
        # errorbar = np.full(output.shape, np.NaN)
        if output_eb:
            errorbar = np.std(outputs, axis=2)
            output = np.hstack([output, errorbar])
        if output_pandas:
            names = self._target_names if output_eb else self._base_target_names
            output = pd.DataFrame(output, columns=names, index=input.index)
        return output

    @property
    def _feature_names(self):
        return self._nns[0]._feature_names if len(self._nns) > 0 else None

    @property
    def _feature_max(self):
        feature_max = None
        if len(self._nns) > 0:
            feature_max = pd.Series(
                np.full_like(self._nns[0]._feature_max, np.inf),
                index=self._nns[0]._feature_max.index,
            )
            for nn in self._nns:
                feature_max = nn._feature_max.combine(feature_max, min)
        return feature_max

    @property
    def _feature_min(self):
        feature_min = None
        if len(self._nns) > 0:
            feature_min = pd.Series(
                np.full_like(self._nns[0]._feature_min, -np.inf),
                index=self._nns[0]._feature_min.index,
            )
            for nn in self._nns:
                feature_min = nn._feature_min.combine(feature_min, max)
        return feature_min

    @property
    def _target_names(self):
        names = None
        if len(self._nns) > 0:
            names = self._nns[0]._target_names
            eb_names = pd.Series([name + "_EB" for name in self._nns[0]._target_names])
            names = names.append(eb_names)
            names = names.reset_index(drop=True)
        return names

    @property
    def _target_max(self):
        target_max = None
        if len(self._nns) > 0:
            eb_names = pd.Series([name + "_EB" for name in self._nns[0]._target_names])
            target_max = pd.Series(
                np.full_like(self._nns[0]._target_max, np.inf),
                index=self._nns[0]._target_max.index,
            )
            for nn in self._nns:
                target_max = nn._target_max.combine(target_max, min)
            eb_max = pd.Series([np.inf] * len(eb_names), index=eb_names)
            target_max = target_max.append(eb_max)
        return target_max

    @property
    def _target_min(self):
        target_min = None
        if len(self._nns) > 0:
            eb_names = pd.Series([name + "_EB" for name in self._nns[0]._target_names])
            target_min = pd.Series(
                np.full_like(self._nns[0]._target_min, -np.inf),
                index=self._nns[0]._target_min.index,
            )
            for nn in self._nns:
                target_min = nn._target_min.combine(target_min, max)
            eb_min = pd.Series([-np.inf] * len(eb_names), index=eb_names)
            target_min = target_min.append(eb_min)
        return target_min

    @property
    def _base_target_names(self):
        return self._nns[0]._target_names


class QuaLiKizCommitteeProductNN:
    """
    Output wrapper which multiplies outputs of two QuaLiKizCommitteeNNs.
    Does not use the native QuaLiKizComboNN due to the special treatment
    requried for the variances.

    :arg target_names: list. Specifies new output column names when using output_pandas option.

    :arg nns: list. QuaLiKizCommitteeNN objects, accepts many but will only apply product to the first two.
    """

    def __init__(self, target_names, nns):
        if len(nns) != 2:
            raise TypeError("QuaLiKizCommitteeProductNN class only accepts two members!")
        self._nns = nns
        if np.any(self._feature_min > self._feature_max):
            raise Exception("Feature min > feature max")

        self._target_names = pd.Series(target_names)
        if not self._target_names.index.is_unique:
            raise Exception("Non unique index for target_names!")
        self._target_min = pd.Series(
            self._combo_func(*[nn._target_min.values for nn in nns]),
            index=self._target_names,
        )
        self._target_max = pd.Series(
            self._combo_func(*[nn._target_max.values for nn in nns]),
            index=self._target_names,
        )

    def _combo_func(self, x, y):
        return x * y

    def _error_combo_func(self, vxidx, vyidx, exidx, eyidx, x, y):
        ycomp = (
            np.power(x[:, vxidx] * y[:, eyidx], 2.0)
            if len(eyidx) > 0
            else np.zeros(x[:, vxidx].shape)
        )
        xcomp = (
            np.power(x[:, exidx] * y[:, vyidx], 2.0)
            if len(exidx) > 0
            else np.zeros(y[:, vyidx].shape)
        )
        return np.sqrt(ycomp + xcomp)

    def get_output(
        self,
        input,
        output_pandas=True,
        clip_low=False,
        clip_high=False,
        low_bound=None,
        high_bound=None,
        safe=True,
        **kwargs,
    ):
        method = kwargs.pop("mismatch_method") if "mismatch_method" in kwargs else "none"
        if not isinstance(method, str):
            method = "none"
        if method not in ["none", "duplicate", "truncate"]:
            method = "none"
        if not safe:
            raise Exception("Only safe mode is supported by %s" % (type(self).__name__))
        clip_x = kwargs.pop("clip_x") if "clip_x" in kwargs else False
        clip_y = kwargs.pop("clip_y") if "clip_y" in kwargs else False
        raw_outputs = []
        if clip_x:
            raw_outputs.append(
                self._nns[0].get_output(
                    input,
                    output_pandas=False,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    low_bound=None,
                    high_bound=None,
                    safe=safe,
                    **kwargs,
                )
            )
        else:
            raw_outputs.append(
                self._nns[0].get_output(
                    input,
                    output_pandas=False,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    low_bound=low_bound,
                    high_bound=high_bound,
                    safe=safe,
                    **kwargs,
                )
            )
        if clip_y:
            raw_outputs.append(
                self._nns[1].get_output(
                    input,
                    output_pandas=False,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    low_bound=None,
                    high_bound=None,
                    safe=safe,
                    **kwargs,
                )
            )
        else:
            raw_outputs.append(
                self._nns[1].get_output(
                    input,
                    output_pandas=False,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    low_bound=low_bound,
                    high_bound=high_bound,
                    safe=safe,
                    **kwargs,
                )
            )

        outputs = []
        prod0_targets = [name for name in self._nns[0]._target_names if not name.endswith("_EB")]
        prod0_indices = []
        for item in prod0_targets:
            prod0_indices.append(pd.Index(self._nns[0]._target_names.values).get_loc(item))
        prod1_targets = [name for name in self._nns[1]._target_names if not name.endswith("_EB")]
        prod1_indices = []
        for item in prod1_targets:
            prod1_indices.append(pd.Index(self._nns[1]._target_names.values).get_loc(item))
        #        if len(prod0_targets) > len(prod1_targets):
        #            if method == 'duplicate':
        #                out0 = raw_outputs[0][:,prod0_indices]
        #                out1 = np.zeros(out0.shape)
        #                if len(prod0_targets) % len(prod1_targets) == 0:
        #                    out1 = raw_outputs[1][:,prod1_indices]
        #                    for ii in range(1,int(len(prod0_targets) / len(prod1_targets))):
        #                        out1 = np.hstack((out1,raw_outputs[1][:,prod1_indices]))
        #                else:
        #                    out1 = raw_outputs[1][:,prod1_indices[0]]
        #                    for ii in range(1,len(prod0_targets)):
        #                        out1 = np.hstack((out1,raw_outputs[1][:,prod1_indices[0]]))
        #                outputs = [out0,out1]
        #            elif method == 'truncate':
        #                out1 = raw_outputs[1][:,prod1
        err0_targets = [name for name in self._nns[0]._target_names if name.endswith("_EB")]
        err0_indices = []
        for item in err0_targets:
            err0_indices.append(pd.Index(self._nns[0]._target_names.values).get_loc(item))
        err1_targets = [name for name in self._nns[1]._target_names if name.endswith("_EB")]
        err1_indices = []
        for item in err1_targets:
            err1_indices.append(pd.Index(self._nns[1]._target_names.values).get_loc(item))

        err_indices = []
        if raw_outputs[0].shape == raw_outputs[1].shape:
            outputs = raw_outputs
            err_indices = err0_indices
        else:
            fullidx = 0 if raw_outputs[0].shape[1] >= raw_outputs[1].shape[1] else 1
            outputs = [
                np.zeros(raw_outputs[fullidx].shape),
                np.zeros(raw_outputs[fullidx].shape),
            ]
            outputs[fullidx] = raw_outputs[fullidx]
            if fullidx == 0:
                outputs[1][:, : raw_outputs[1].shape[1]] = raw_outputs[1]
                err_indices = err0_indices
            else:
                outputs[0][:, : raw_outputs[0].shape[1]] = raw_outputs[0]
                err_indices = err1_indices

        output = self._combo_func(*outputs)
        if len(err0_targets) > 0 or len(err1_targets) > 0:
            errorbars = self._error_combo_func(
                prod0_indices, prod1_indices, err0_indices, err1_indices, *outputs
            )
            output[:, err_indices] = errorbars
        if output_pandas:
            output = pd.DataFrame(output, columns=self._target_names, index=input.index)
        return output

    @property
    def _feature_names(self):
        return self._nns[0]._feature_names

    @property
    def _feature_max(self):
        feature_max = pd.Series(
            np.full_like(self._nns[0]._feature_max, np.inf),
            index=self._nns[0]._feature_max.index,
        )
        for nn in self._nns:
            feature_max = nn._feature_max.combine(feature_max, min)
        return feature_max

    @property
    def _feature_min(self):
        feature_min = pd.Series(
            np.full_like(self._nns[0]._feature_min, -np.inf),
            index=self._nns[0]._feature_min.index,
        )
        for nn in self._nns:
            feature_min = nn._feature_min.combine(feature_min, max)
        return feature_min


class QuaLiKizNDNNCollection(QuaLiKizNDNN):
    """
    Wrapper for sets of QuaLiKizNDNN objects whose outputs have
    unique names. Primarily for user convenience as it condenses
    the large number of NNs into a single object.

    Currently does not provide easy functions for adding or removing
    NNs to the collection.

    :arg nns: list. Set of NNs with unique output names to place into collection.
    """

    def __init__(self, nns):
        self._nns = nns
        if not self._feature_names.index.is_unique:
            raise Exception("Non unique index for feature_names!")
        if np.any(self._feature_min > self._feature_max):
            raise Exception("Feature min > feature max")
        if not self._target_names.index.is_unique:
            raise Exception("Non unique index for target_names!")

    def get_output(
        self,
        input,
        output_pandas=True,
        clip_low=False,
        clip_high=False,
        low_bound=None,
        high_bound=None,
        safe=True,
        **kwargs,
    ):
        outlist = [
            nn.get_output(
                input,
                output_pandas=output_pandas,
                clip_low=clip_low,
                clip_high=clip_high,
                low_bound=low_bound,
                high_bound=high_bound,
                safe=safe,
                **kwargs,
            )
            for nn in self._nns
        ]
        output = None
        if output_pandas:
            output = outlist[0]
            for nn in outlist[1:]:
                output = output.join(nn)
        else:
            output = np.hstack(outlist)
        return output

    @property
    def _feature_names(self):
        feature_names = pd.Series()
        for nn in self._nns:
            feature_names = feature_names.append(nn._feature_names)
        feature_names = pd.Series(feature_names.unique())
        return feature_names

    @property
    def _feature_max(self):
        feature_max = pd.Series(
            np.full_like(self._nns[0]._feature_max, np.inf),
            index=self._nns[0]._feature_max.index,
        )
        for nn in self._nns:
            feature_max = nn._feature_max.combine(feature_max, min)
        return feature_max

    @property
    def _feature_min(self):
        feature_min = pd.Series(
            np.full_like(self._nns[0]._feature_min, -np.inf),
            index=self._nns[0]._feature_min.index,
        )
        for nn in self._nns:
            feature_min = nn._feature_min.combine(feature_min, max)
        return feature_min

    @property
    def _target_names(self):
        target_names = pd.Series()
        for nn in self._nns:
            target_names = target_names.append(nn._target_names)
        target_names = pd.Series(target_names.unique())
        return target_names

    @property
    def _target_max(self):
        target_max = pd.Series(
            np.full_like(self._nns[0]._target_max, np.inf),
            index=self._nns[0]._target_max.index,
        )
        for nn in self._nns:
            target_max = nn._target_max.combine(target_max, min)
        return target_max

    @property
    def _target_min(self):
        target_min = pd.Series(
            np.full_like(self._nns[0]._target_min, np.inf),
            index=self._nns[0]._target_min.index,
        )
        for nn in self._nns:
            target_min = nn._target_min.combine(target_min, max)
        return target_min


if __name__ == "__main__":
    scann = 100

    root = os.path.dirname(os.path.realpath(__file__))
    nn1 = QuaLiKizNDNN.from_json(
        "../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json",
        layer_mode="classic",
    )
    nn2 = QuaLiKizNDNN.from_json(
        "../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json",
        layer_mode="classic",
    )
    nn = QuaLiKizCommitteeNN([nn1, nn2])

    scann = 100
    input = pd.DataFrame()
    input["Ati"] = np.array(np.linspace(2, 13, scann))
    input["Ti_Te"] = np.full_like(input["Ati"], 1.0)
    input["Zeff"] = np.full_like(input["Ati"], 1.0)
    input["An"] = np.full_like(input["Ati"], 2.0)
    input["Ate"] = np.full_like(input["Ati"], 5.0)
    input["q"] = np.full_like(input["Ati"], 0.660156)
    input["smag"] = np.full_like(input["Ati"], 0.399902)
    input["Nustar"] = np.full_like(input["Ati"], 0.009995)
    input["logNustar"] = np.full_like(input["Ati"], np.log10(0.009995))
    input["x"] = np.full_like(input["Ati"], 0.449951)
    # input = input.loc[:, nn_ITG._feature_names]
    input["Machtor"] = np.full_like(input["Ati"], 0.3)
    fluxes = nn.get_output(input, safe=True)
    print(fluxes)
