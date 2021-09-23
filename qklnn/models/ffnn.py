#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import os
from warnings import warn
from collections import OrderedDict

import numpy as np
import pandas as pd
from IPython import embed


def sigm_tf(x):
    return 1.0 / (1 + np.exp(-1 * x))


# def sigm(x):
#    return 2./(1 + np.exp(-2 * x)) - 1


def flatten(l):
    return [item for sublist in l for item in sublist]


def nn_dict_to_matlab(json_file):
    newjs = {}
    for key, val in json_file.items():
        newjs[key.replace("/", "_").replace(":", "_")] = val
    matdict = {", ".join(newjs["target_names"]): newjs}
    return matdict


class QuaLiKizComboNN:
    def __init__(self, target_names, nns, combo_func):
        self._nns = nns
        for nn in self._nns:
            if np.any(nn._feature_names.ne(self._feature_names)):
                raise Exception("Supplied NNs have different feature names")
        if np.any(self._feature_min > self._feature_max):
            raise Exception("Feature min > feature max")

        self._combo_func = combo_func
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
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = determine_settings(
            self, input, safe, clip_low, clip_high, low_bound, high_bound
        )
        output = self._combo_func(
            *[
                nn.get_output(
                    nn_input,
                    output_pandas=False,
                    clip_low=False,
                    clip_high=False,
                    safe=safe,
                    **kwargs,
                )
                for nn in self._nns
            ]
        )
        output = clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound)
        if output_pandas is True:
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


class QuaLiKizSelfComboNN:
    """
    Network output wrapper which applies specified operation on network outputs
    upon evaluation. Differs from QuaLiKizComboNN in that this class applies
    the operation to across a single multi-output NN instead of across multiple
    single-output NNs.

    :arg target_names: list. Specifies new output column names when using output_pandas option.

    :arg nn: QuaLiKizNDNN. Neural network, this class only accepts a single network object.

    :arg combo_func: callable. Operation to apply to NN outputs, can accept any number of arguments as long as it is reflected in indices.

    :arg indices: list. Specifies which of the original column names are passed to the operation function, for each new output column.
    """

    def __init__(self, target_names, nn, combo_func, indices):
        self._nn = nn
        if np.any(self._feature_min > self._feature_max):
            raise Exception("Feature min > feature max")

        self._combo_func = combo_func
        self._target_names = pd.Series(target_names)
        if not self._target_names.index.is_unique:
            raise Exception("Non unique index for target_names!")
        for index in range(0, len(indices)):
            for item in indices[index]:
                if item is not None and item not in self._nn._target_names.values:
                    raise Exception("Requested operation on non-existant target_name!")
        self._combo_indices = indices
        if len(self._combo_indices) != len(self._target_names):
            raise Exception("Number of target names and operations do not match")
        target_min = []
        target_max = []
        for index in range(0, len(self._combo_indices)):
            if None in self._combo_indices[index]:
                target_min.append(self._nn._target_min[self._combo_indices[index][0]])
                target_max.append(self._nn._target_max[self._combo_indices[index][0]])
            else:
                target_min.append(
                    self._combo_func(
                        *[self._nn._target_min[name] for name in self._combo_indices[index]]
                    )
                )
                target_max.append(
                    self._combo_func(
                        *[self._nn._target_max[name] for name in self._combo_indices[index]]
                    )
                )
        self._target_min = pd.Series(target_min, index=self._target_names)
        self._target_max = pd.Series(target_max, index=self._target_names)

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
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = determine_settings(
            self, input, safe, clip_low, clip_high, low_bound, high_bound
        )
        pre_output = self._nn.get_output(
            nn_input, output_pandas=True, clip_low=False, clip_high=False, safe=safe, **kwargs
        )
        eval_out = []
        for index in range(0, len(self._combo_indices)):
            if None in self._combo_indices[index]:
                eval_out.append(pre_output[self._combo_indices[index][0]].values)
            else:
                eval_out.append(
                    self._combo_func(
                        *[pre_output[name].values for name in self._combo_indices[index]]
                    )
                )
        output = np.hstack([np.transpose(np.atleast_2d(item)) for item in eval_out])
        output = clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound)
        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names, index=input.index)
        return output

    @property
    def _feature_names(self):
        return self._nn._feature_names

    @property
    def _feature_max(self):
        return self._nn._feature_max

    @property
    def _feature_min(self):
        return self._nn._feature_min


class QuaLiKizNDNN:
    def __init__(self, nn_dict, target_names_mask=None, layer_mode=None, GB_scale_length=1):
        """General ND fully-connected multilayer perceptron neural network

        Initialize this class using a nn_dict. This dict is usually read
        directly from JSON, and has a specific structure. Generate this JSON
        file using the supplied function in QuaLiKiz-Tensorflow
        """
        parsed = {}
        if layer_mode is None:
            try:
                import qlknn_intel
            except:
                layer_mode = "classic"
            else:
                layer_mode = "intel"
        elif layer_mode == "intel":
            import qlknn_intel
        elif layer_mode == "cython":
            import cython_mkl_ndnn
        self.GB_scale_length = GB_scale_length

        # Read and parse the json. E.g. put arrays in arrays and the rest in a dict
        for name, value in nn_dict.items():
            if name == "hidden_activation" or name == "output_activation":
                parsed[name] = value
            elif value.__class__ == list:
                parsed[name] = np.array(value)
            else:
                parsed[name] = dict(value)
        # These variables do not depend on the amount of layers in the NN
        for set in ["feature", "target"]:
            setattr(self, "_" + set + "_names", pd.Series(parsed.pop(set + "_names")))
        for set in ["feature", "target"]:
            for subset in ["min", "max"]:
                setattr(
                    self,
                    "_".join(["", set, subset]),
                    pd.Series(parsed.pop("_".join([set, subset])))[
                        getattr(self, "_" + set + "_names")
                    ],
                )
        for subset in ["bias", "factor"]:
            setattr(
                self,
                "_".join(["_feature_prescale", subset]),
                pd.Series(parsed["prescale_" + subset])[self._feature_names],
            )
            setattr(
                self,
                "_".join(["_target_prescale", subset]),
                pd.Series(parsed.pop("prescale_" + subset))[self._target_names],
            )
        self.layers = []
        # Now find out the amount of layers in our NN, and save the weigths and biases
        activations = parsed["hidden_activation"] + [parsed["output_activation"]]
        for ii in range(1, len(activations) + 1):
            try:
                name = "layer" + str(ii)
                weight = parsed.pop(name + "/weights/Variable:0")
                bias = parsed.pop(name + "/biases/Variable:0")
                activation = activations.pop(0)
                if layer_mode == "classic":
                    if activation == "tanh":
                        act = np.tanh
                    elif activation == "relu":
                        act = _act_relu
                    elif activation == "none":
                        act = _act_none
                    self.layers.append(QuaLiKizNDNN.NNLayer(weight, bias, act))
                elif layer_mode == "intel":
                    self.layers.append(qlknn_intel.Layer(weight, bias, activation))
                elif layer_mode == "cython":
                    self.layers.append(cython_mkl_ndnn.Layer(weight, bias, activation))
            except KeyError:
                # This name does not exist in the JSON,
                # so our previously read layer was the output layer
                break
        if len(activations) == 0:
            del parsed["hidden_activation"]
            del parsed["output_activation"]
        try:
            self._clip_bounds = parsed["_metadata"]["clip_bounds"]
        except KeyError:
            self._clip_bounds = False

        self._target_names_mask = target_names_mask
        # Ignore metadata
        try:
            self._metadata = parsed.pop("_metadata")
        except KeyError:
            pass
        # Ignore parsed settings
        try:
            self._parsed_settings = parsed.pop("_parsed_settings")
        except KeyError:
            pass
        if any(parsed):
            warn("nn_dict not fully parsed! " + str(parsed))

    def apply_layers(self, input, output=None):
        """Apply all NN layers to the given input

        The given input has to be array-like, but can be of size 1
        """
        input = np.ascontiguousarray(input)
        # 3x30 network:
        # 14.1 µs ± 913 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        # 20.9 µs ± 2.43 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        # 19.1 µs ± 240 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        # 2.67 µs ± 29.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

        for layer in self.layers:
            output = np.empty([input.shape[0], layer._weights.shape[1]])
            output = layer.apply(input, output)
            input = output
        return input

    class NNLayer:
        """A single (hidden) NN layer
        A hidden NN layer is just does

        output = activation(weight * input + bias)

        Where weight is generally a matrix; output, input and bias a vector
        and activation a (sigmoid) function.
        """

        def __init__(self, weight, bias, activation):
            self._weights = weight
            self._biases = bias
            self._activation = activation

        def apply(self, input, output=None):
            preactivation = np.dot(input, self._weights) + self._biases
            result = self._activation(preactivation)
            return result

        def shape(self):
            return self.weight.shape

        def __str__(self):
            return "NNLayer shape " + str(self.shape())

    def get_output(
        self,
        input,
        clip_low=False,
        clip_high=False,
        low_bound=None,
        high_bound=None,
        safe=True,
        output_pandas=True,
        **kwargs,
    ):
        """Calculate the output given a specific input

        This function accepts inputs in the form of a dict with
        as keys the name of the specific input variable (usually
        at least the feature_names) and as values 1xN same-length
        arrays.
        """
        # 49.1 ns ± 1.53 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = determine_settings(
            self, input, safe, clip_low, clip_high, low_bound, high_bound
        )

        # nn_input = self._feature_prescale_factors.values[np.newaxis, :] * nn_input + self._feature_prescale_biases.values
        # 14.3 µs ± 1.08 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        nn_input = _prescale(
            nn_input,
            self._feature_prescale_factor.values,
            self._feature_prescale_bias.values,
        )

        # Apply all NN layers an re-scale the outputs
        # 104 µs ± 19.7 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        # 70.9 µs ± 384 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each) (only apply layers)
        output = (
            self.apply_layers(nn_input) - np.atleast_2d(self._target_prescale_bias)
        ) / np.atleast_2d(self._target_prescale_factor)
        # for name in self._target_names:
        #    nn_output = (np.squeeze(self.apply_layers(nn_input)) - self._target_prescale_biases[name]) / self._target_prescale_factors[name]
        #    output[name] = nn_output
        scale_mask = [
            not any(prefix in name for prefix in ["df", "chie", "xaxis"])
            for name in self._target_names
        ]
        if self.GB_scale_length != 1 and any(scale_mask):
            output[:, scale_mask] /= self.GB_scale_length
        output = clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound)

        # 118 µs ± 3.83 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        if output_pandas:
            output = pd.DataFrame(output, columns=self._target_names, index=input.index)

        # 47.4 ns ± 1.79 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        if self._target_names_mask is not None:
            output.columns = self._target_names_mask
        return output

    @classmethod
    def from_json(cls, json_file, **kwargs):
        with open(json_file) as file_:
            dict_ = json.load(file_)
        nn = cls(dict_, **kwargs)
        return nn

    @property
    def l2_norm(self):
        l2_norm = 0
        for layer in self.layers:
            l2_norm += np.sum(np.square(layer.weight))
        l2_norm /= 2
        return l2_norm

    @property
    def l1_norm(self):
        l1_norm = 0
        for layer in self.layers:
            l1_norm += np.sum(np.abs(layer.weight))
        return l1_norm


class QuaLiKizLessDNN(QuaLiKizNDNN):
    def __init__(
        self,
        nn_dict,
        const_dict=None,
        Zi=None,
        target_names_mask=None,
        layer_mode=None,
        set_all_An_equal=True,
        set_all_Ati_equal=True,
        set_all_Ti_Te_equal=True,
        normni_from_zeff=True,
    ):
        self.set_all_An_equal = set_all_An_equal
        self.set_all_Ati_equal = set_all_Ati_equal
        self.set_all_Ti_Te_equal = set_all_Ti_Te_equal
        self.normni_from_zeff = normni_from_zeff
        self._internal_network = QuaLiKizNDNN(
            nn_dict, target_names_mask=target_names_mask, layer_mode=layer_mode
        )
        self._feature_names = self._internal_network._feature_names
        self._target_names = self._internal_network._target_names
        self._Zi = Zi
        self._feature_min = self._internal_network._feature_min
        self._feature_max = self._internal_network._feature_max
        self._target_min = self._internal_network._target_min
        self._target_max = self._internal_network._target_max
        for varname in ["An", "Ati", "Ti_Te"]:
            if getattr(self, "set_all_" + varname + "_equal") and any(
                name.startswith(varname) for name in self._internal_network._feature_names
            ):
                is_subgroup = self._feature_names.apply(lambda x: x.startswith(varname))
                setattr(self, "_" + varname + "_vars", self._feature_names.loc[is_subgroup])
                subvars = getattr(self, "_" + varname + "_vars")
                self._feature_names = self._feature_names.drop(subvars.index)
                self._feature_names = self._feature_names.append(
                    pd.Series(varname, index=[self._feature_names.index.max() + 1])
                )
                for var, op in {"min": np.max, "max": np.min}.items():
                    subvals = getattr(self, "_feature_" + var)[subvars]
                    setattr(
                        self,
                        "_feature_" + var,
                        getattr(self, "_feature_" + var).drop(subvars),
                    )
                    setattr(
                        self,
                        "_feature_" + var,
                        getattr(self, "_feature_" + var).append(
                            pd.Series(op(subvals), index=[varname])
                        ),
                    )

        if self.normni_from_zeff and any(
            name.startswith("normni") for name in self._internal_network._feature_names
        ):
            if not "Zeff" in self._internal_network._feature_names.values:
                raise Exception("normni_from_zeff is True, but network does not depend on Zeff")
            self._normni_vars = self._feature_names.loc[
                self._feature_names.apply(lambda x: x.startswith("normni"))
            ]
            if len(self._normni_vars) > 2:
                raise Exception(
                    "normni_from_zeff assumes two ions, but network depends on one, three or more"
                )
            if self._Zi is None or len(self._Zi) != 2:
                raise Exception("normni_from_zeff is True, but no Zi of length two given")
            self._feature_names = self._feature_names.drop(self._normni_vars.index)
            self._feature_min = self._feature_min.drop(self._normni_vars)
            self._feature_max = self._feature_max.drop(self._normni_vars)

        self._const_dict = const_dict
        for varname in self._const_dict:
            self._feature_names = self._feature_names.drop(
                self._feature_names[self._feature_names == varname].index
            )

    def get_output(
        self,
        input,
        clip_low=False,
        clip_high=False,
        low_bound=None,
        high_bound=None,
        safe=True,
        output_pandas=True,
        **kwargs,
    ):
        """Calculate the output given a specific input

        This function accepts inputs in the form of a dict with
        as keys the name of the specific input variable (usually
        at least the feature_names) and as values 1xN same-length
        arrays.
        """
        if not isinstance(input, pd.DataFrame) or not safe:
            raise NotImplementedError("Unsafe mode or non-DataFrame input")
        if clip_low or clip_high:
            warn(
                "Clipping of in/output not implemented in %s" % (type(self).__name__),
                UserWarning,
            )

        old_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        for varname in ["An", "Ati", "Ti_Te"]:
            if varname in self._feature_names.values and getattr(
                self, "set_all_" + varname + "_equal"
            ):
                if varname in input:
                    for name in getattr(self, "_" + varname + "_vars"):
                        input.loc[:, name] = input.loc[:, varname]
                else:
                    raise KeyError("{!s} not in input".format(varname))

        if "Zeff" in self._internal_network._feature_names.values and self.normni_from_zeff:
            normni0, normni1 = calculate_normni(self._Zi[0], self._Zi[1], input.loc[:, "Zeff"])
            input.loc[:, "normni0"] = normni0
            input.loc[:, "normni1"] = normni1

        for varname, val in self._const_dict.items():
            input.loc[:, varname] = val
        pd.options.mode.chained_assignment = old_setting
        fluxes = QuaLiKizNDNN.get_output(
            self._internal_network,
            input,
            clip_low=clip_low,
            clip_high=clip_high,
            low_bound=low_bound,
            high_bound=high_bound,
            safe=safe,
            output_pandas=output_pandas,
        )
        return fluxes


def calculate_normni(Z0, Z1, Zeff):
    normni1 = (Zeff - Z0) / (Z1 ** 2 - Z1 * Z0)
    normni0 = (1 - Z1 * normni1) / Z0
    return normni0, normni1


def clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound):
    if clip_low:
        if isinstance(low_bound, (int, float)):
            output[output < low_bound] = low_bound
        else:
            for ii, bound in enumerate(low_bound):
                output[:, ii][output[:, ii] < bound] = bound

    if clip_high:
        if isinstance(high_bound, (int, float)):
            output[output < high_bound] = high_bound
        else:
            for ii, bound in enumerate(high_bound):
                output[:, ii][output[:, ii] > bound] = bound

    return output


def determine_settings(network, input, safe, clip_low, clip_high, low_bound, high_bound):
    if safe:
        if isinstance(input, pd.DataFrame):
            nn_input = input[network._feature_names]
        else:
            raise Exception("Please pass a pandas.DataFrame for safe mode")
        if low_bound is not None:
            low_bound = (
                low_bound
                if isinstance(low_bound, (int, float))
                else low_bound.loc[network._target_names].values
            )
        if high_bound is not None:
            high_bound = (
                high_bound
                if isinstance(high_bound, (int, float))
                else high_bound.loc[network._target_names].values
            )
    else:
        if input.__class__ == pd.DataFrame:
            nn_input = input.values
        elif input.__class__ == np.ndarray:
            nn_input = input

    if clip_low is True and (low_bound is None):
        low_bound = network._target_min.values
    if clip_high is True and (high_bound is None):
        high_bound = network._target_max.values
    return nn_input, safe, clip_low, clip_high, low_bound, high_bound


def _prescale(nn_input, factors, biases):
    return np.atleast_2d(factors) * nn_input + np.atleast_2d(biases)


def _act_none(x):
    return x


def _act_relu(x):
    return x * (x > 0)


if __name__ == "__main__":
    # Test the function
    root = os.path.dirname(os.path.realpath(__file__))
    # nn1 = QuaLiKizNDNN.from_json(os.path.join(root, 'nn_efe_GB.json'))
    # nn2 = QuaLiKizNDNN.from_json(os.path.join(root, 'nn_efi_GB.json'))
    # nn = QuaLiKizMultiNN([nn1, nn2])
    nn_path = os.path.join(root, "../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json")
    nn = QuaLiKizNDNN.from_json(nn_path)

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
    input["x"] = np.full_like(input["Ati"], 0.449951)
    input["logNustar"] = np.full_like(input["Ati"], 1e-3)
    input["Zeff"] = np.full_like(input["Ati"], 1)
    input["dilution"] = np.full_like(input["Ati"], 1.0)
    input = input[nn._feature_names]

    fluxes = nn.get_output(input)
    # fluxes = nn.get_output(input.values, safe=False)
    print(fluxes)

    try:
        nn2 = QuaLiKizNDNN.from_json(nn_path, layer_mode="intel")
        fluxes2 = nn2.get_output(input.values, safe=False)
        print(fluxes2)
    except Exception as ee:
        print("Problem loading intel style:")
        print(ee)

    try:
        nn3 = QuaLiKizNDNN.from_json(nn_path, layer_mode="cython")
        fluxes3 = nn2.get_output(input.values, safe=False)
        print(fluxes3)
    except Exception as ee:
        print("Problem loading cython style:")
        print(ee)

    nn_div_path = os.path.join(
        root, "../../tests/gen3_test_files/Network_302_efeITG_GB_div_efiITG_GB/nn.json"
    )
    nn_div = QuaLiKizNDNN.from_json(nn_div_path)
    nn_combo = QuaLiKizComboNN(["efeITG_GB"], [nn, nn_div], lambda x, y: x * y)
    fluxes4 = nn_combo.get_output(input)
    print(fluxes4)
