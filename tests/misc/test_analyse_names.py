import pytest
from IPython import embed  # noqa: F401

from qlknn.misc.analyse_names import *


def is_transport_helper(name):
    # Try if we can split in parts
    parts = split_parts(name)
    # Try if we can extract parts
    extracted = extract_part_names(parts)
    # Try if we can split the name
    for part_name in extracted:
        splitted = split_name(part_name)  # noqa: F841
    # Can we deterime if this is transport
    transp = is_transport(name)
    return transp


class TestPrimitives:
    simple_testnames = [
        "efe_GB",
        "efi_SI",
        "pfeETG_GB",
        "verITG_SI",
        "chieTEM_SI",
    ]
    simple_parts = {
        "efe_GB": ["efe_GB"],
        "efi_SI": ["efi_SI"],
        "pfeETG_GB": ["pfeETG_GB"],
        "verITG_SI": ["verITG_SI"],
        "chieTEM_SI": ["chieTEM_SI"],
    }

    @pytest.mark.parametrize("name", simple_testnames)
    def test_simple_name_splitting(self, name):
        splitted = split_parts(name)
        assert len(splitted) == 1
        assert splitted == self.simple_parts[name]

    combined_testnames = [
        "efe_GB_div_efi_GB",
        "efiITG_GB_fake_chiei_GB",
        "efiITG_GB_fake_chiei_GB_triple_pfe_SI",
    ]
    combined_parts = {
        "efe_GB_div_efi_GB": ["efe_GB", "_div_", "efi_GB"],
        "efiITG_GB_fake_chiei_GB": ["efiITG_GB", "_fake_", "chiei_GB"],
        "efiITG_GB_fake_chiei_GB_triple_pfe_SI": [
            "efiITG_GB",
            "_fake_",
            "chiei_GB",
            "_triple_",
            "pfe_SI",
        ],
    }

    @pytest.mark.parametrize("name", combined_testnames)
    def test_combined_name_splitting(self, name):
        splitted = split_parts(name)
        assert splitted == self.combined_parts[name]

    combined_part_names = {
        "efe_GB_div_efi_GB": ["efe_GB", "efi_GB"],
        "efiITG_GB_fake_chiei_GB": ["efiITG_GB", "chiei_GB"],
        "efiITG_GB_fake_chiei_GB_triple_pfe_SI": ["efiITG_GB", "chiei_GB", "pfe_SI"],
    }

    @pytest.mark.parametrize("name", combined_testnames)
    def test_combined_name_part_names(self, name):
        splitted = split_parts(name)
        extracted = extract_part_names(splitted)
        assert extracted == self.combined_part_names[name]

    def test_part_names_exception(self):
        with pytest.raises(TypeError) as excinfo:
            extract_part_names("efe_GB")
        assert "iven a string" in str(excinfo.value)

    combined_operations = {
        "efe_GB_div_efi_GB": ["_div_"],
        "efiITG_GB_fake_chiei_GB": ["_fake_"],
        "efiITG_GB_fake_chiei_GB_triple_pfe_SI": ["_fake_", "_triple_"],
    }

    @pytest.mark.parametrize("name", combined_testnames)
    def test_combined_name_operations(self, name):
        splitted = split_parts(name)
        extracted = extract_operations(splitted)
        assert extracted == self.combined_operations[name]

    def test_operations_exception(self):
        with pytest.raises(TypeError) as excinfo:
            extract_operations("efe_GB")
        assert "iven a string" in str(excinfo.value)


class TestAnalysis:
    test_names = [
        "pfe_GB",
        "pfeETG_SI",
        "pfeITG_SI_div_efeETG_GB",
        "pfeITG_SI_div_pfeTEM_GB",
        "efe_GB",
        "efe_GB_plus_efiITG_GB",
        "efeTEM_GB_plus_efiTEM_GB",
        "efeTEM_GB_plus_efi_GB",
        "efeITG_GB_plus_efiITG_GB",
        "efeITG_GB_plus_efi_GB",
        "efeETG_GB_plus_efiETG_GB",
        "efeETG_GB_plus_efi_GB",
        "chiee_GB",
        "gam_GB",
    ]

    is_transport = {
        "pfe_GB": True,
        "pfeETG_SI": True,
        "pfeITG_SI_div_efeETG_GB": True,
        "pfeITG_SI_div_pfeTEM_GB": True,
        "efe_GB": True,
        "efe_GB_plus_efiITG_GB": True,
        "efeTEM_GB_plus_efiTEM_GB": True,
        "efeTEM_GB_plus_efi_GB": True,
        "efeITG_GB_plus_efiITG_GB": True,
        "efeITG_GB_plus_efi_GB": True,
        "efeETG_GB_plus_efiETG_GB": True,
        "efeETG_GB_plus_efi_GB": True,
        "chiee_GB": True,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_transport(self, name):
        transp = is_transport_helper(name)
        assert transp == self.is_transport[name]

    is_partial_particle = {
        "pfe_GB": True,
        "pfeETG_SI": True,
        "pfeITG_SI_div_efeETG_GB": True,
        "pfeITG_SI_div_pfeTEM_GB": True,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_partial_particle(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_partial_particle(name) == self.is_partial_particle[name]

    is_pure_heat = {
        "pfe_GB": False,
        "pfeETG_SI": False,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": True,
        "efe_GB_plus_efiITG_GB": True,
        "efeTEM_GB_plus_efiTEM_GB": True,
        "efeTEM_GB_plus_efi_GB": True,
        "efeITG_GB_plus_efiITG_GB": True,
        "efeITG_GB_plus_efi_GB": True,
        "efeETG_GB_plus_efiETG_GB": True,
        "efeETG_GB_plus_efi_GB": True,
        "chiee_GB": True,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_pure_heat(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_pure_heat(name) == self.is_pure_heat[name]

    is_tem_scale = {
        "pfe_GB": False,
        "pfeETG_SI": False,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": True,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_tem_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_mode_scale(name, "tem") == self.is_tem_scale[name]

    is_itg_scale = {
        "pfe_GB": False,
        "pfeETG_SI": False,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": True,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_itg_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_mode_scale(name, "itg") == self.is_itg_scale[name]

    is_etg_scale = {
        "pfe_GB": False,
        "pfeETG_SI": True,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": True,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_etg_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_mode_scale(name, "etg") == self.is_etg_scale[name]

    is_ion_scale = {
        "pfe_GB": False,
        "pfeETG_SI": False,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": True,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": True,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": True,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_ion_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_ion_scale(name) == self.is_ion_scale[name]

    is_electron_scale = {
        "pfe_GB": False,
        "pfeETG_SI": True,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": False,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": True,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_electron_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_electron_scale(name) == self.is_electron_scale[name]

    is_multi_scale = {
        "pfe_GB": True,
        "pfeETG_SI": False,
        "pfeITG_SI_div_efeETG_GB": True,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": True,
        "efe_GB_plus_efiITG_GB": True,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": True,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": True,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": True,
        "chiee_GB": True,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_multi_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_multi_scale(name) == self.is_multi_scale[name]

    is_pure_flux = {
        "pfe_GB": True,
        "pfeETG_SI": True,
        "pfeITG_SI_div_efeETG_GB": False,
        "pfeITG_SI_div_pfeTEM_GB": False,
        "efe_GB": True,
        "efe_GB_plus_efiITG_GB": False,
        "efeTEM_GB_plus_efiTEM_GB": False,
        "efeTEM_GB_plus_efi_GB": False,
        "efeITG_GB_plus_efiITG_GB": False,
        "efeITG_GB_plus_efi_GB": False,
        "efeETG_GB_plus_efiETG_GB": False,
        "efeETG_GB_plus_efi_GB": False,
        "chiee_GB": False,
        "gam_GB": False,
    }

    @pytest.mark.parametrize("name", test_names)
    def test_is_pure_flux(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert is_pure_flux(name) == self.is_pure_flux[name]

    special_input = {
        "pfe_GB": [],
        "pfeETG_SI": ["Ate"],
        "pfeITG_SI_div_efeETG_GB": [],
        "pfeITG_SI_div_pfeTEM_GB": [],
        "efe_GB": [],
        "efe_GB_plus_efiITG_GB": [],
        "efeTEM_GB_plus_efiTEM_GB": ["Ate"],
        "efeTEM_GB_plus_efi_GB": [],
        "efeITG_GB_plus_efiITG_GB": ["Ati"],
        "efeITG_GB_plus_efi_GB": [],
        "efeETG_GB_plus_efiETG_GB": ["Ate"],
        "efeETG_GB_plus_efi_GB": [],
        "chiee_GB": [],
        "gam_GB": [],
    }

    @pytest.mark.parametrize("name", test_names)
    def test_determine_special_input(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert determine_special_input(name) == self.special_input[name]

    mode_scale = {
        "pfe_GB": "unknown",
        "pfeETG_SI": "ETG",
        "pfeITG_SI_div_efeETG_GB": "unknown",
        "pfeITG_SI_div_pfeTEM_GB": "unknown",
        "efe_GB": "unknown",
        "efe_GB_plus_efiITG_GB": "unknown",
        "efeTEM_GB_plus_efiTEM_GB": "TEM",
        "efeTEM_GB_plus_efi_GB": "unknown",
        "efeITG_GB_plus_efiITG_GB": "ITG",
        "efeITG_GB_plus_efi_GB": "unknown",
        "efeETG_GB_plus_efiETG_GB": "ETG",
        "efeETG_GB_plus_efi_GB": "unknown",
        "chiee_GB": "unknown",
        "gam_GB": "unknown",
    }

    @pytest.mark.parametrize("name", test_names)
    def test_determine_mode_scale(self, name):
        is_transport_helper(name)  # Can we deterime if this is transport
        assert determine_mode_scale(name) == self.mode_scale[name]
