import re
import warnings

from IPython import embed

particle_rotationless_diffusion_vars = ["df", "vt", "vc"]
particle_rotation_diffusion_vars = ["vr"]
particle_diffusion_vars = particle_rotationless_diffusion_vars + particle_rotation_diffusion_vars
particle_flux = ["pf"]
particle_vars = particle_flux + particle_diffusion_vars
heat_rotationless_diffusion_vars = ["chie", "ven", "vec"]
heat_rotation_diffusion_vars = ["ver"]
heat_diffusion_vars = heat_rotationless_diffusion_vars + heat_rotation_diffusion_vars
heat_flux = ["ef"]
heat_vars = heat_flux + heat_diffusion_vars
momentum_flux = ["vf"]
momentum_vars = momentum_flux
rotation_vars = particle_rotation_diffusion_vars
transport_coeff_pattern = "(?:[a-zA-Z]{2,4})(?:[a-zA-Z])(?:|ITG|ETG|TEM)_(?:GB|SI|cm)"


def split_parts(name):
    """Split combined variable name into parts

    Tries to split a name into parts. For example, efeETG_GB_div_efiITG_SI into
    efeETG, _div_, and efiITG_SI. Specifically, it assumes the pattern
    QuaLiKiz-transport-coefficient-name and the optionally an infinitely
    repeating pattern of operation name + QuaLiKiz-transport-coefficient-name.

    Args:
        name: The name to be split in parts

    Returns:
        List containing the split-off parts
    """
    splitted = re.compile("(" + transport_coeff_pattern + ")").split(name)
    if splitted[0] != "" or splitted[-1] != "":
        raise ValueError("Split {!s} in an unexpected way: {!s}".format(name, splitted))
    del splitted[0], splitted[-1]
    return splitted


def extract_part_names(splitted):
    """Extract the QuaLiKiz-transport-coefficient-names from list of parts"""
    if isinstance(splitted, str):
        raise TypeError("Given a string, should be given a list-like object")
    return splitted[slice(0, len(splitted), 2)]


def extract_operations(splitted):
    """Extract the operations from list of parts"""
    if isinstance(splitted, str):
        raise TypeError("Given a string, should be given a list-like object")
    return splitted[slice(1, len(splitted) - 1, 2)]


def is_pure(name):
    """ Test if the name is pure, e.g. contains no operations """
    try:
        pure = len(split_parts(name)) == 1
    except ValueError:
        pure = False
    return pure


def is_flux(name):
    """Test if all parts of a name are fluxes

    For example, efe_GB, efe_GB_div_efi_GB are all composed of fluxes, but
    gam_GB and chiee_GB_div_efi_GB are not.
    """
    flux = True
    try:
        for part_name in extract_part_names(split_parts(name)):
            flux &= split_name(part_name)[0] in heat_flux + particle_flux + momentum_flux
    except ValueError:
        flux = False
    return flux


def is_transport(name):
    """Test if all parts of a name are transport coefficients

    For example, efe_GB, chie_GB_div_efi_GB are all composed of transport
    coefficients, but gam_GB and chiee_GB_plus_gam_GB are not.
    """
    transport = True
    try:
        for part_name in extract_part_names(split_parts(name)):
            transport &= split_name(part_name)[0] in heat_vars + particle_vars + momentum_vars
    except ValueError:
        transport = False
    return transport


def is_pure_flux(name):
    """Test if name is a pure flux

    For example, efe_GB is a pure flux, chie_GB and efe_GB_div_efi_GB are not.
    """
    pure_flux = is_pure(name) and is_flux(name)
    return pure_flux


def split_name(name):
    """Split a transport-coefficient-like name into parts

    This tries to extract the transport coefficient type, species, mode, and
    normalization. For example efe_GB will be split in ['ef', 'e', '', 'GB']
    end chieiETG_SI will be split in ['chie', 'i', 'ETG', 'SI']. If the name
    cannot be split, return the full name
    """
    splitted = re.compile(transport_coeff_pattern.replace("?:", "")).split(name)
    if splitted[0] != "" or splitted[-1] != "":
        raise ValueError("Split {!s} in an unexpected way: {!s}".format(name, splitted))
    del splitted[0], splitted[-1]
    # Splitted should be of the form ['vc', 'i', 'ITG', 'GB'] now
    try:
        transp, species, mode, norm = splitted
        return transp, species, mode, norm
    except ValueError:
        return name


def is_growth(name):
    """ Test if name is a growth rate-like variable"""
    return name in ["gam_leq_GB", "gam_great_GB"]


def contains_sep(name):
    """ Test if name contains a mode name, e.g. TEM, ITG, ETG"""
    return any(sub in name for sub in ["TEM", "ITG", "ETG"])


def is_full_transport(name):
    """Test if name as a total/full transport channel

    For example, efe_GB and chiee_GB both are total transport,
    but efeETG_GB and gam_GB are not.
    """
    return is_transport(name) and not contains_sep(name)


def is_transport_family(name, identifiers, combiner):
    """Check if a name is part of a family of transport coefficients

    It performs the check by splitting the name in parts. Then it is checked
    that the first part contains any of the identifiers. For the rest of the
    parts (if there are any) the same check is performed, and combined with
    the given combiner function with the previous result(s). Common choices
    are `x and y` if _all_ parts should contain one of the identifiers, or
    `x or y` if _any_ part should contain one of the identifiers.

    Args:
        name: The name to be analyzed
        identifiers: List of name-parts that need to be checked for. If any
                     part is in the name, the name is part of the transport family
        combiner: Function determining how the internal boolean checks relate.
    """
    if isinstance(identifiers, str):
        raise TypeError("Identifiers is a string, should be given a list-like object")
    if is_transport(name):
        parts = split_parts(name)
        part_names = extract_part_names(parts)
        transport_family = any(sub in part_names[0] for sub in identifiers)
        for part in part_names[1:]:
            transport_family = combiner(transport_family, any(sub in part for sub in identifiers))
    else:
        transport_family = False
    return transport_family


def is_pure_diffusion(name):
    return is_transport_family(name, particle_diffusion_vars, lambda x, y: x and y)


def is_pure_heat(name):
    return is_transport_family(name, heat_vars, lambda x, y: x and y)


def is_pure_particle(name):
    return is_transport_family(name, particle_vars, lambda x, y: x and y)


def is_pure_rot(name):
    return is_transport_family(name, rotation_vars, lambda x, y: x and y)


def is_pure_momentum(name):
    return is_transport_family(name, momentum_vars, lambda x, y: x and y)


def is_partial_diffusion(name):
    return is_transport_family(name, particle_diffusion_vars, lambda x, y: x or y)


def is_partial_heat(name):
    return is_transport_family(name, heat_vars, lambda x, y: x or y)


def is_partial_particle(name):
    return is_transport_family(name, particle_vars, lambda x, y: x or y)


def is_partial_rot(name):
    return is_transport_family(name, rotation_vars, lambda x, y: x or y)


def is_partial_momentum(name):
    return is_transport_family(name, momentum_vars, lambda x, y: x or y)


def is_mode_scale(name, mode):
    """Check if name is all from the same mode scale

    Checks if all parts of the name match the given mode

    Args:
      - name: name to be analysed. Can contain operators, e.g. efe_GB_div_efi_GB
      - mode: mode to search for in parts. e.g. TEM, ITG

    Returns:
      - True if all parts are of the same mode. False if not
    """
    return is_transport_family(name, [mode.upper()], lambda x, y: x and y)


def is_ion_scale(name):
    if is_transport(name):
        parts = split_parts(name)
        part_names = extract_part_names(parts)
        transp, species, mode, norm = split_name(part_names[0])
        is_scale = not any(sub == mode for sub in ["", "ETG"])
        for part_name in part_names[1:]:
            transp, species, mode, norm = split_name(part_name)
            is_scale = is_scale and not any(sub == mode for sub in ["", "ETG"])
    else:
        is_scale = False
    return is_scale


def is_electron_scale(name):
    return is_mode_scale(name, "ETG")


def is_multi_scale(name):
    return not is_electron_scale(name) and not is_ion_scale(name) and is_transport(name)


def determine_driving_gradients(name):
    """Determine the gradients driving the given named variable

    Can only happen on pure-moded variable names, see `is_mode_scale`.
    Returns a list of driving gradients ordered on perceived importance.
    As this is subjective, do not count on specific ordering for wrapping
    codes. Returns an empty list if a driving gradient could not be determined,
    for example for mixed-mode names

    Args:
     - name: name to be analysed. Can contain operators, e.g. efe_GB_div_efi_GB

    Returns:
     - driving_gradient: A list of driving gradients
    """
    # TODO: Do we want multi-driving gradients (e.g. Ane etc.)?
    if is_mode_scale(name, "ETG") or is_mode_scale(name, "TEM"):
        # Is pure ETG, or is pure TEM
        driving_gradient = ["Ate"]
    elif is_mode_scale(name, "ITG"):
        # Is pure ITG
        driving_gradient = ["Ati"]
    else:
        driving_gradient = []
    return driving_gradient


def determine_special_input(name):
    return determine_driving_gradients(name)


def determine_mode_scale(name):
    """Determine if a given flux is 'ITG', 'TEM' or 'ETG' mode

    Args:
     - name: name to be analysed. Can contain operators, e.g. efe_GB_div_efi_GB
    """
    mode_scale = "unknown"
    for mode in ["ITG", "TEM", "ETG"]:
        if is_mode_scale(name, mode):
            mode_scale = mode
            break
    return mode_scale


def is_leading(name):
    if is_transport(name):
        leading = True
        if not is_full_transport(name):
            if any(sub in name for sub in ["div", "plus"]):
                if "ITG" in name:
                    if name not in [
                        "efeITG_GB_div_efiITG_GB",
                        "pfeITG_GB_div_efiITG_GB",
                    ]:
                        leading = False
                elif "TEM" in name:
                    if name not in [
                        "efiTEM_GB_div_efeTEM_GB",
                        "pfeTEM_GB_div_efeTEM_GB",
                    ]:
                        leading = False
            if "pfi" in name:
                leading = False
    else:
        leading = False
    return leading


if __name__ == "__main__":
    print(split_parts("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB"))
    print(split_parts("efeITG_GB_div_efiITG_GB"))
    print(split_parts("efeITG_GB"))

    print(extract_part_names(split_parts("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB")))
    print(extract_part_names(split_parts("efeITG_GB_div_efiITG_GB")))
    print(extract_part_names(split_parts("efeITG_GB")))

    print(extract_operations(split_parts("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB")))
    print(extract_operations(split_parts("efeITG_GB_div_efiITG_GB")))
    print(extract_operations(split_parts("efeITG_GB")))

    print(is_pure("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB"))
    print(is_pure("efeITG_GB_div_efiITG_GB"))
    print(is_pure("efeITG_GB"))

    print(is_pure_flux("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB"))
    print(is_pure_flux("efeITG_GB_div_efiITG_GB"))
    print(is_pure_flux("efeITG_GB"))
    print(is_pure_heat("efiITG_GB"))

    print(is_flux("efeITG_GB_div_efiITG_GB_plus_pfeITG_GB"))
    print(is_flux("efeITG_GB_div_efiITG_GB"))
    print(is_flux("efeITG_GB"))
