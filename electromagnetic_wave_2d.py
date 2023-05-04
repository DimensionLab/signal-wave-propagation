import numpy as np
from sympy import Symbol, Function, Number, sqrt

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.geometry.primitives_2d import Rectangle, Line
from modulus.key import Key
#from modulus.eq.pdes.wave_equation import WaveEquation
from modulus.eq.pde import PDE
from modulus.utils.io.plotter import ValidatorPlotter

from modulus.models.modified_fourier_net import ModifiedFourierNetArch

import math

# Read in npz files generated using finite difference simulator Devito
def read_wf_data(time, dLen):
    wf_filename = to_absolute_path(f"/root/Desktop/workspace/examples/seismic_wave/Training_data/wf_{int(time):04d}ms.npz")
    wave = np.load(wf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, wave.shape[0]),
        np.linspace(0, dLen, wave.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 0.001)
    outvar = {}
    outvar["u"] = np.expand_dims(wave.flatten(), axis=-1)
    return invar, outvar





class WavePlotter(ValidatorPlotter):
    "Define custom validator plotting class"

    def __call__(self, invar, true_outvar, pred_outvar):

        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs


@modulus.main(config_path="conf", config_name="config_el")
def run(cfg: ModulusConfig) -> None:
    """
    2d wave propagation at a domain of 2kmx3km, with a single Ricker source at the middle of the 2D domain 
    Impedance BC comes between air and concrete parts 
    interior has different impedance and attenuation constants for each medium 
    """

    """
    For air, the approximate values are:

        Permittivity (ε_air): 8.854 * 10⁻¹² F/m
        Permeability (μ_air): 4π * 10⁻⁷ H/m
        Conductivity (σ_air): close to 0 S/m (air is a poor conductor)
    For homogeneous concrete, the values can vary depending on the specific mixture, but typical values are:

        Permittivity (ε_concrete): ~ 7 * 8.854 * 10⁻¹² F/m
        Permeability (μ_concrete): 4π * 10⁻⁷ H/m (assuming no significant magnetic materials)
        Conductivity (σ_concrete): ~ 0.01 - 1 S/m (depends on the specific mixture)
    """

    #define constants
    # air
    mu1 = 4 * math.pi * 1e-7
    eps1 = 8.854 * 1e-12
    sigma1 = 0.1 * 1e-12
    # concrete 
    mu2 = 4 * math.pi * 1e-7
    eps2 = 7 * 8.854 * 1e-12
    sigma2 = 0.1

    f = 5 * 1e9  # frequency of the signal, for now 5GHz

    # alpha is the attenuation constant of a medium
    # α = sqrt(ω² * μ * ε / 2) * sqrt(1 + (σ / (ω * ε))^2 - 1)
    def countAlpha(f=1, mu=1, eps=1, sigma=1):
        omega = 2 * math.pi * f
        helper = sqrt(1 + (pow(sigma, 2) / (pow(eps, 2) * pow(omega, 2))))
        a = omega * sqrt((helper * mu * eps) / 2 - 1)
        return a

    alpha1 = countAlpha(f, mu1, eps1, sigma1)
    alpha2 = countAlpha(f, mu2, eps2, sigma2)

    # calculate the speed of light in given medium
    c1 = 1 / math.sqrt(mu1 * eps1)
    c2 = 1 / math.sqrt(mu2 * eps2)

    # wave numbers
    # wn1 = 

    # override defaults
    # cfg.arch.fully_connected.layer_size = 128

    # define PDEs
    # wave equation for first medium, speed c and attenuation are different than in we2
    we = WaveEquationAtten(u="u", c=c1, dim=2, time=True, alpha=alpha1)
    # wave equation for second medium
    we2 = WaveEquationAtten(u="u", c=c2, dim=2, time=True, alpha=alpha2)
    # open boundary shoud be provided by c="c", if done so, it crashes. This has to be solved later
    """
    could not unroll graph!
    This is probably because you are asking to compute a value that is not an output of any node
    ####################################
    invar: [x, y, normal_x, normal_y, area, t]
    requested var: [open_boundary]
    computable var: [x, y, normal_x, normal_y, area, t, wave_equation, u, ux, uy, wave_equation, ImpedanceBC_reflected_x, ImpedanceBC_reflected_y, ImpedanceBC_transmitted_x, ImpedanceBC_transmitted_y]
    """
    ob = OpenBoundary(u="u", c=c1, dim=2, time=True) 
    # impedance boundary is described within its definition
    ib = ImpedanceBC(ux="ux", uy="uy", mu1=mu1, mu2=mu2, epsilon1=eps1, epsilon2=eps2, dimension=2, time=True)

    # define networks and nodes

    # Modified fourier net had following issues:
    # If OpenBoundary is left out - error appears: loss went to Nans
    # Could not unroll graph, possibly similar to the above  
    """  
    wave_net = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("ux"), Key("uy")],
        frequencies=("axis, diagonal", [f]),
        frequencies_params=("axis, diagonal", [f]),
        #layer_size=params["layer_size"],
        #nr_layers=params["nr_layers"],
    )
    """
    
    """
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("ux"), Key("uy")],
        frequencies=("axis,diagonal", [f]),
        frequencies_params=(
            "axis,diagonal",
            [f],
        ),
        cfg=cfg.arch.modified_fourier,
    )
    """
    # if we use fully connected network, there is no change in loss, see example:
    """
    07:16:48] - [step:       5100] loss:  1.998e+23, time/iteration:  3.059e+01 ms
    [07:16:50] - [step:       5200] loss:  2.393e+23, time/iteration:  2.192e+01 ms
    [07:16:52] - [step:       5300] loss:  2.138e+23, time/iteration:  2.180e+01 ms
    [07:16:55] - [step:       5400] loss:  2.049e+23, time/iteration:  2.175e+01 ms
    """
    # we do not change the speed of wave this time (unlike wave_2d example)
    
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("ux"), Key("uy")],
        cfg=cfg.arch.fully_connected,
    )
    
    # nodes are - wave eq. 1, wave eq. 2, open boundary and integral boundary
    # we want to find the wave amplitude at given location in given time
    # it is unclear to me if detach or not, c is problem now
    nodes = (
        we.make_nodes(detach_names=["c"])
        + we2.make_nodes(detach_names=["c"])
        + ob.make_nodes(detach_names=["c"])
        + ib.make_nodes(detach_names=["c"])  # detach_names=["c"]
        + [
            wave_net.make_node(name="wave_network"),
            # speed_net.make_node(name="speed_network"),
        ]
    )

    # define geometry
    # now just a simple 2D geo - two parts representing 1x1km of air on left and 1x1km of concrete on right (hella lot of concrete, wow)
    dLen = 1  # km
    rec_air = Rectangle((0, 0), (dLen, dLen))
    rec_concrete = Rectangle((dLen, 0), (2 * dLen, dLen))
    rec = rec_air + rec_concrete
    boundary_line = Line((dLen, 0), (dLen, dLen))
    # in future I'll add buildings
    # building = Rectangle((1,1.5), (0.3,0.3))

    # define sympy domain variables - x, y for coords, t for time 
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

    # define time range
    time_length = 1
    # isn't this too much? maybe we need lower time range - frequency is very high
    time_range = {t: (0.001, time_length)}

    """
    # we do not change the speed here, the amplitude is what we care about
    # commenting out following section from example
    # define target velocity model
    # 2.0 km/s at the bottom and 1.0 km/s at the top using tanh function
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, 2, 512), np.linspace(0, 2, 512), indexing="ij"
    )
    wave_speed_invar = {}
    wave_speed_invar["x"] = np.expand_dims(mesh_x.flatten(), axis=-1)
    wave_speed_invar["y"] = np.expand_dims(mesh_y.flatten(), axis=-1)
    wave_speed_outvar = {}
    wave_speed_outvar["c"] = np.tanh(80 * (wave_speed_invar["y"] - 1.0)) / 2 + 1.5
    """

    # make domain
    domain = Domain()
    """
    # no velocity is changed in electromagnetic wave
    # add velocity constraint
    velocity = PointwiseConstraint.from_numpy(
        nodes=nodes, invar=wave_speed_invar, outvar=wave_speed_outvar, batch_size=1024
    )
    domain.add_constraint(velocity, "Velocity")
    """
    # add initial timesteps constraints
    # Ricker source is probably loaded here as well
    batch_size = 1024
    for i, ms in enumerate(np.linspace(150, 300, 4)):
        timestep_invar, timestep_outvar = read_wf_data(ms, dLen)
        lambda_weighting = {}
        lambda_weighting["u"] = np.full_like(timestep_invar["x"], 10.0 / batch_size)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        #domain.add_constraint(timestep, f"BC{i:04d}")

    #  here lambda weighting might cause problem!

    # add interior constraint - left side of interior = air
    interior_left = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec_air,
        outvar={"wave_equation": 0},
        batch_size=1024,
        # bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"wave_equation": 0.0001},
        parameterization=time_range,
    )
    domain.add_constraint(interior_left, "Interior_left")

    # add interior constraint - right side of interior = concrete
    interior_right = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec_concrete,
        outvar={"wave_equation": 0},
        batch_size=1024,
        #bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"wave_equation": 0.0001},
        parameterization=time_range,
    )
    domain.add_constraint(interior_right, "Interior_right")

    # add open boundary constraint - around the whole domain
    edges = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"open_boundary": 0},
        batch_size=256,
        lambda_weighting={"open_boundary": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(edges, "Edges")
    
    # add impedance boundary constraint - between left and right interior
    # reflected wave - x
    reflx = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_reflected_x": 0},
        batch_size=256,
        lambda_weighting={"ImpedanceBC_reflected_x": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(reflx, "Refl_x")

    # reflected wave - y
    refly = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_reflected_y": 0},
        batch_size=256,
        lambda_weighting={"ImpedanceBC_reflected_y": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(refly, "Refl_y")

    """
    reflz = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_reflected_z": 0},
        batch_size=1024,
        lambda_weighting={"ImpedanceBC_reflected_z": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(reflz, "Refl_z")"""

    # transmitted wave - x
    transx = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_transmitted_x": 0},
        batch_size=256,
        lambda_weighting={"ImpedanceBC_transmitted_x": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(transx, "Trans_x")

    # transmitted wave - y
    transy = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_transmitted_y": 0},
        batch_size=256,
        lambda_weighting={"ImpedanceBC_transmitted_y": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(transy, "Trans_y")

    """
    transz = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=boundary_line,
        outvar={"ImpedanceBC_transmitted_z": 0},
        batch_size=1024,
        lambda_weighting={"ImpedanceBC_transmitted_z": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(transz, "Trans_z")"""

    # add validators
    """for i, ms in enumerate(np.linspace(350, 950, 13)):
        val_invar, val_true_outvar = read_wf_data(ms, dLen)
        validator = PointwiseValidator(
            nodes=nodes,
            invar=val_invar,
            true_outvar=val_true_outvar,
            batch_size=1024,
            plotter=WavePlotter(),
        )
        domain.add_validator(validator, f"VAL_{i:04d}")
    validator = PointwiseValidator(
        nodes=nodes,
        invar=wave_speed_invar,
        true_outvar=wave_speed_outvar,
        batch_size=1024,
        plotter=WavePlotter(),
    )
    domain.add_validator(validator, "Velocity")
    """

    slv = Solver(cfg, domain)

    slv.solve()


class ImpedanceBC(PDE):
    """
    Impedance Boundary Condition for Electromagnetic Waves
    Equations: 
        A_r = A * (z_1 - z_2) / (z_1 + z_2)
        A_t = A * (2 * z_1) / (z_1 + z_2)
    Parameters
    ==========
    ux : str
       Ex
    uy : str
       Ey
    uz : str
       Ez
    mu1 : float
        Permeability of medium 1
    mu2 : float
        Permeability of medium 2
    epsilon1 : float
        Permittivity of medium 1
    epsilon2 : float
        Permittivity of medium 2
    dimension : int
        Dimension of the problem (2 or 3)
    """

    name = "ImpedanceBC"

    def __init__(self, ux="ux", uy="uy", uz="uz", c="c", mu1=1, mu2=1, epsilon1=1, epsilon2=1, dimension=2, time=True):
        assert dimension in (2, 3), "dimension must be 2 or 3"

        # set params
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.dim = dimension
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}

        # decide which variables to keep
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # E field - actually amplitude of the wave
        assert isinstance(ux, str), "ux needs to be string"
        ux = Function(ux)(*input_variables)

        assert isinstance(uy, str), "uy needs to be string"
        uy = Function(uy)(*input_variables)

        assert isinstance(uz, str), "uz needs to be string"
        uz = Function(uz)(*input_variables)


        # compute the impedance for each medium
        z1 = sqrt(mu1 / epsilon1)
        z2 = sqrt(mu2 / epsilon2)

        # normal vector
        n = [normal_x, normal_y]

        # incident E field vector
        E_incident = [self.ux, self.uy]
        if self.dim == 3:
            E_incident.append(self.uz)
            n.append(normal_z)

        E_parallel = []
        # parallel component of the E field
        for v, n in zip(E_incident, n):
            E_parallel.append(Symbol(v) * n)
        #E_parallel = normal.dot.vec(E_incident, n) * n

        # perpendicular component of the E field
        print(E_incident[0], E_parallel[0])
        E_perpendicular = [Symbol(E_incident[i]) - E_parallel[i] for i in range(dimension)]

        # reflected and transmitted amplitudes for perpendicular component
        A_r = (z1 - z2) / (z1 + z2)
        A_t = (2 * z1) / (z1 + z2)

        # compute the reflected and transmitted fields
        E_reflected = [A_r * E_perpendicular[i] for i in range(dimension)]
        E_transmitted = [A_t * E_perpendicular[i] for i in range(dimension)]

        # set the boundary conditions for each component of the E field
        self.equations = {}
        for i, component in enumerate(['x', 'y', 'z'][:dimension]):
            self.equations[f"ImpedanceBC_reflected_{component}"] = E_reflected[i]
            self.equations[f"ImpedanceBC_transmitted_{component}"] = E_transmitted[i]


class WaveEquationAtten(PDE):
    """
    Wave equation with attenuation

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the wave equation.
    alpha : float
        Attenuation coefficient, a constant defining the medium attenuation to the wave propagation

    Examples
    ========
    >>> we = WaveEquation(c=0.8, dim=3)
    >>> we.pprint()
      wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    >>> we = WaveEquation(c='c', dim=2, time=False)
    >>> we.pprint()
      wave_equation: -c**2*u__x__x - c**2*u__y__y - 2*c*c__x*u__x - 2*c*c__y*u__y
    """

    name = "WaveEquation"

    def __init__(self, u="u", c="c", dim=3, time=True, mixed_form=False, alpha=0.1):
        # set params
        self.u = u
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        self.alpha = alpha  # attenuation added atop of classical wave equation

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}

        if not self.mixed_form:
            self.equations["wave_equation"] = (
                u.diff(t, 2)
                - c**2 * u.diff(x, 2)
                - c**2 * u.diff(y, 2)
                - c**2 * u.diff(z, 2)
                - 2 * self.alpha * u.diff(t)
            )
        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            if self.dim == 3:
                u_z = Function("u_z")(*input_variables)
            else:
                u_z = Number(0)
            if self.time:
                u_t = Function("u_t")(*input_variables)
            else:
                u_t = Number(0)

            self.equations["wave_equation"] = (
                u_t.diff(t)
                - c**2 * u_x.diff(x)
                - c**2 * u_y.diff(y)
                - c**2 * u_z.diff(z)
                - 2 * self.alpha * u_t
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            if self.dim == 2:
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")

                # define open boundary conditions
class OpenBoundary(PDE):
    """
    Open boundary condition for wave problems
    Ref: http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
    """

    name = "OpenBoundary"

    def __init__(self, u="u", c="c", dim=2, time=True):
        # set params
        self.u = u
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # normal
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["open_boundary"] = (
            u.diff(t)
            + normal_x * c * u.diff(x)
            + normal_y * c * u.diff(y)
            + normal_z * c * u.diff(z)
        )


if __name__ == "__main__":
    run()
