//! `hklrs.InverseError` and one subclass per `hklrs::InverseError` variant,
//! every one raised through [`to_py`]. Angles they carry are degrees.

use hklrs::InverseError as E;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::Degrees;

create_exception!(hklrs, InverseError, PyValueError, "The inverse failed.");
create_exception!(
    hklrs,
    Unreachable,
    InverseError,
    "|Q| exceeds 4 pi / wavelength; carries q and q_max (Å⁻¹)."
);
create_exception!(
    hklrs,
    OffEwaldSphere,
    InverseError,
    "No written axis turns the sample or the beam and the target is off the Ewald sphere; \
     carries q (sample frame) and miss, |k_i + Q| - k (Å⁻¹)."
);
create_exception!(
    hklrs,
    NoSolution,
    InverseError,
    "No axis values satisfy the mode."
);
create_exception!(
    hklrs,
    NoSolutionInLimits,
    NoSolution,
    "Every solution takes an axis outside its limits; carries them as \
     solutions (degrees / metres, nearest to current first)."
);
create_exception!(
    hklrs,
    GeometryError,
    InverseError,
    "The geometry rejected the mode or the axis values, as an axis the mode \
     names that the geometry lacks."
);
create_exception!(
    hklrs,
    Degenerate,
    InverseError,
    "The mode does not isolate the solutions: a continuum or a multiple root."
);
create_exception!(
    hklrs,
    PsiUndefined,
    InverseError,
    "psi is undefined: Q, kf x ki or the reference normal to Q is null."
);
create_exception!(
    hklrs,
    NoKappaArm,
    InverseError,
    "A kappa equation or target on a mode without a kappa arm."
);
create_exception!(
    hklrs,
    NotKappaArm,
    InverseError,
    "The named axes do not form a kappa arm; carries names."
);
create_exception!(
    hklrs,
    KappaAlpha,
    InverseError,
    "Kappa tilt outside (0, 90]; carries alpha (degrees)."
);
create_exception!(
    hklrs,
    ChiOutOfReach,
    InverseError,
    "|chi| exceeds the kappa reach 2 alpha; carries chi and reach (degrees)."
);
create_exception!(
    hklrs,
    EquationCount,
    InverseError,
    "Written axes and equations differ in number; carries written and equations."
);
create_exception!(
    hklrs,
    NonFinite,
    InverseError,
    "An input is not finite; carries what."
);
create_exception!(
    hklrs,
    ZeroReference,
    InverseError,
    "A reference vector is zero; carries what."
);
create_exception!(
    hklrs,
    NoSuchAxis,
    InverseError,
    "An axis index outside the geometry; carries index and count."
);
create_exception!(
    hklrs,
    NotRotation,
    InverseError,
    "The axis is not a rotation; carries name."
);
create_exception!(
    hklrs,
    WrittenTwice,
    InverseError,
    "The mode writes an axis twice; carries name."
);
create_exception!(
    hklrs,
    NotInStack,
    InverseError,
    "The axis is not in the holder's stack; carries name and holder."
);
create_exception!(
    hklrs,
    SlitsMoveDetector,
    InverseError,
    "The slits axis moves the detector; carries name."
);
create_exception!(
    hklrs,
    NotPreset,
    InverseError,
    "The geometry is not a preset and has no named modes; carries geometry."
);
create_exception!(
    hklrs,
    UnknownMode,
    InverseError,
    "The geometry's preset has no mode of this name; carries geometry and name."
);
create_exception!(
    hklrs,
    MissingParameter,
    InverseError,
    "The mode needs a parameter it was not given; carries mode and parameter."
);
create_exception!(
    hklrs,
    UnknownParameter,
    InverseError,
    "The mode takes no parameter of this name; carries mode and parameter."
);
create_exception!(
    hklrs,
    NotSwitch,
    InverseError,
    "The mode takes this parameter as 0 or 1 alone; carries mode, parameter and value."
);
create_exception!(
    hklrs,
    EmergenceRange,
    InverseError,
    "Emergence outside [-90, 90]; carries emergence (degrees)."
);

/// `e` as the subclass named after its variant. `geometry` is the one the
/// failing call solved on, for the units of `NoSolutionInLimits.solutions`.
pub(crate) fn to_py(py: Python<'_>, geometry: &hklrs::Geometry, e: E) -> PyErr {
    // the message in degrees; the attributes stay exact
    let msg = match &e {
        E::Geometry(g) => crate::message(g.clone()),
        E::EmergenceRange { emergence } => {
            format!("emergence {}° is outside [-90°, 90°]", Degrees(*emergence))
        }
        E::KappaAlpha { alpha } => {
            format!(
                "kappa arm tilt alpha = {}° is outside (0°, 90°]",
                Degrees(*alpha)
            )
        }
        E::ChiOutOfReach { chi, reach } => format!(
            "|chi| = {}° exceeds the kappa reach 2·alpha = {}°",
            Degrees(chi.abs()),
            Degrees(*reach)
        ),
        e => e.to_string(),
    };
    // Plain values cannot fail to set on a fresh exception.
    macro_rules! raise {
        ($exc:ident $(, $k:literal => $v:expr)*) => {{
            let e = $exc::new_err(msg);
            $(let _ = e.value(py).setattr($k, $v);)*
            e
        }};
    }
    match e {
        E::Unreachable { q, q_max } => raise!(Unreachable, "q" => q, "q_max" => q_max),
        E::OffEwaldSphere { q, miss } => raise!(OffEwaldSphere, "q" => q, "miss" => miss),
        E::NoSolution => raise!(NoSolution),
        E::NoSolutionInLimits(outside) => match outside
            .iter()
            .map(|s| geometry.values_to_degrees(s))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(solutions) => raise!(NoSolutionInLimits, "solutions" => solutions),
            Err(g) => to_py(py, geometry, E::Geometry(g)),
        },
        E::Geometry(_) => raise!(GeometryError),
        E::Degenerate => raise!(Degenerate),
        E::PsiUndefined => raise!(PsiUndefined),
        E::NoKappaArm => raise!(NoKappaArm),
        E::NotKappaArm { names } => raise!(NotKappaArm, "names" => names),
        E::KappaAlpha { alpha } => raise!(KappaAlpha, "alpha" => alpha.to_degrees()),
        E::ChiOutOfReach { chi, reach } => raise!(
            ChiOutOfReach,
            "chi" => chi.to_degrees(),
            "reach" => reach.to_degrees()
        ),
        E::EquationCount { written, equations } => raise!(
            EquationCount,
            "written" => written,
            "equations" => equations
        ),
        E::NonFinite { what } => raise!(NonFinite, "what" => what),
        E::ZeroReference { what } => raise!(ZeroReference, "what" => what),
        E::NoSuchAxis { index, count } => {
            raise!(NoSuchAxis, "index" => index, "count" => count)
        }
        E::NotRotation { name } => raise!(NotRotation, "name" => name),
        E::WrittenTwice { name } => raise!(WrittenTwice, "name" => name),
        E::NotInStack { name, holder } => {
            raise!(NotInStack, "name" => name, "holder" => holder)
        }
        E::SlitsMoveDetector { name } => raise!(SlitsMoveDetector, "name" => name),
        E::EmergenceRange { emergence } => {
            raise!(EmergenceRange, "emergence" => emergence.to_degrees())
        }
        E::NotPreset { geometry } => raise!(NotPreset, "geometry" => geometry),
        E::UnknownMode { geometry, name } => {
            raise!(UnknownMode, "geometry" => geometry, "name" => name)
        }
        E::MissingParameter { mode, parameter } => {
            raise!(MissingParameter, "mode" => mode, "parameter" => parameter)
        }
        E::UnknownParameter { mode, parameter } => {
            raise!(UnknownParameter, "mode" => mode, "parameter" => parameter)
        }
        E::NotSwitch {
            mode,
            parameter,
            value,
        } => raise!(NotSwitch, "mode" => mode, "parameter" => parameter, "value" => value),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("InverseError", py.get_type::<InverseError>())?;
    m.add("Unreachable", py.get_type::<Unreachable>())?;
    m.add("OffEwaldSphere", py.get_type::<OffEwaldSphere>())?;
    m.add("NoSolution", py.get_type::<NoSolution>())?;
    m.add("NoSolutionInLimits", py.get_type::<NoSolutionInLimits>())?;
    m.add("GeometryError", py.get_type::<GeometryError>())?;
    m.add("Degenerate", py.get_type::<Degenerate>())?;
    m.add("PsiUndefined", py.get_type::<PsiUndefined>())?;
    m.add("NoKappaArm", py.get_type::<NoKappaArm>())?;
    m.add("NotKappaArm", py.get_type::<NotKappaArm>())?;
    m.add("KappaAlpha", py.get_type::<KappaAlpha>())?;
    m.add("ChiOutOfReach", py.get_type::<ChiOutOfReach>())?;
    m.add("EquationCount", py.get_type::<EquationCount>())?;
    m.add("NonFinite", py.get_type::<NonFinite>())?;
    m.add("ZeroReference", py.get_type::<ZeroReference>())?;
    m.add("NoSuchAxis", py.get_type::<NoSuchAxis>())?;
    m.add("NotRotation", py.get_type::<NotRotation>())?;
    m.add("WrittenTwice", py.get_type::<WrittenTwice>())?;
    m.add("NotInStack", py.get_type::<NotInStack>())?;
    m.add("SlitsMoveDetector", py.get_type::<SlitsMoveDetector>())?;
    m.add("EmergenceRange", py.get_type::<EmergenceRange>())?;
    m.add("NotPreset", py.get_type::<NotPreset>())?;
    m.add("UnknownMode", py.get_type::<UnknownMode>())?;
    m.add("MissingParameter", py.get_type::<MissingParameter>())?;
    m.add("UnknownParameter", py.get_type::<UnknownParameter>())?;
    m.add("NotSwitch", py.get_type::<NotSwitch>())
}
