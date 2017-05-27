
# Univ. Michigan Aerostructural model.
# Based on OpenAeroStruct by John Hwang, and John Jasa (github.com/mdolab/OpenAeroStruct)
# author: Sam Friedman  (samfriedman@tamu.edu)
# date:   4/12/2017

"""
analysis.py

This module contains wrapper functions for each part of the multidisciplinary
analysis of the OpenAeroStruct model. Specifically, this is the
solve_nonlinear() method to each OpenMDAO component in OpenAeroStruct. To use
them, first call the setup() function, which returns an OASProblem object. This
object contains the following attributes:

    OASProblem.prob_dict :   Dictionary of problem parameters
    OASProblem.surfaces  :   List of surface dictionaries defining properties of
                                each lifting surface
    OASProblem.comp_dict :   Dictionary of OpenAeroStruct component objects
                                which contain the analysis of each with a
                                dictionary of problem parameters

For each wrapper function, optionally pass in the necessary component object
from the comp_dict dictionary. Using pre-initialized components drastically
reduces the computation time for a full multidisciplinary analysis. Without
pre-initialization of the component, another argument must be given to initialize
the component within the function. This extra argument is usually the surface
dictionary, but can be other problem or surface parameters. An example with
pre-initiazation is shown in aerodynamics() and structures(). A example without
pre-initialization is shown in aerodynamics2() and structures2().

An example of the multidisciplinary analysis of the coupled system is in the
if __name__=="__main__" function. It uses fixed point iteration to converge the
coupled system of loads and displacements.

Current list of function wrappers available:
    vlm_geometry
    assemble_aic
    aero_circulations
    vlm_forces
    vlm_coeffs
    vlm_lift_drag
    total_lift
    total_drag
    compute_nodes
    assemble_k
    spatial_beam_fem
    spatial_beam_disp
    materials_tube
    geometry_mesh
    transfer_displacements
    transfer_loads
    spatialbeam_energy
    spatialbeam_weight
    spatialbeam_failure_ks
    spatialbeam_failure_exact
    functional_breguet_range
    functional_equilibrium

For now, these functions only support a single lifting surface, and does not
support B-spline customization of the lifting surface.

Future work required:
    - Extend functions to be used with multiple lifting surfaces
"""

# make compatible Python 2.x to 3.x
from __future__ import print_function, division
# from future.builtins import range  # make compatible Python 2.x to 3.x

import numpy as np
import math

from materials import MaterialsTube
from spatialbeam import ComputeNodes, AssembleK, SpatialBeamFEM, SpatialBeamDisp, SpatialBeamEnergy, SpatialBeamWeight, SpatialBeamVonMisesTube, SpatialBeamFailureExact, SpatialBeamFailureKS
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMGeometry, AssembleAIC, AeroCirculations, VLMForces, VLMLiftDrag, VLMCoeffs, TotalLift, TotalDrag, ViscousDrag
from geometry import GeometryMesh, Bspline#, MonotonicConstraint
from run_classes import OASProblem
from openmdao.api import Component, Problem, Group
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

try:
    from pprint import PrettyPrinter
    pp = PrettyPrinter(width=1, indent=2)
    def PRINT(msg, var, pp=pp):
        print(msg)
        pp.pprint(var)
except:
    def PRINT(msg, var, pp=None):
        print(msg)
        print(var)

# to disable OpenMDAO warnings which will create an error in Matlab
import warnings
warnings.filterwarnings("ignore")

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

"""
================================================================================
                            GEOMETRY / SETUP
================================================================================
From run_classes.py: Manipulate geometry mesh based on high-level design parameters """


def setup(prob_dict={}, surfaces=[{}]):
    ''' Setup the aerostruct mesh

    Default wing mesh (single lifting surface):
    -------------------------------------------
    name = 'wing'            # name of the surface
    num_x = 3                # number of chordwise points
    num_y = 5                # number of spanwise points
    root_chord = 1.          # root chord
    span_cos_spacing = 1     # 0 for uniform spanwise panels
                             # 1 for cosine-spaced panels
                             # any value between 0 and 1 for a mixed spacing
    chord_cos_spacing = 0.   # 0 for uniform chordwise panels
                             # 1 for cosine-spaced panels
                             # any value between 0 and 1 for a mixed spacing
    wing_type = 'rect'       # initial shape of the wing either 'CRM' or 'rect'
                             # 'CRM' can have different options after it, such as 'CRM:alpha_2.75' for the CRM shape at alpha=2.75
    offset = np.array([0., 0., 0.]) # coordinates to offset the surface from its default location
    symmetry = True          # if true, model one half of wing reflected across the plane y = 0
    S_ref_type = 'wetted'    # 'wetted' or 'projected'

    # Simple Geometric Variables
    span = 10.               # full wingspan
    dihedral = 0.            # wing dihedral angle in degrees positive is upward
    sweep = 0.               # wing sweep angle in degrees positive sweeps back
    taper = 1.               # taper ratio; 1. is uniform chord

    # B-spline Geometric Variables. The number of control points for each of these variables can be specified in surf_dict
    # by adding the prefix "num" to the variable (e.g. num_twist)
    twist_cp = None
    chord_cp = None
    xshear_cp = None
    zshear_cp = None
    thickness_cp = None

    Default wing parameters:
    ------------------------
    Zero-lift aerodynamic performance
        CL0 = 0.0            # CL value at AoA (alpha) = 0
        CD0 = 0.0            # CD value at AoA (alpha) = 0
    Airfoil properties for viscous drag calculation
        k_lam = 0.05         # percentage of chord with laminar flow, used for viscous drag
        t_over_c = 0.12      # thickness over chord ratio (NACA0012)
        c_max_t = .303       # chordwise location of maximum (NACA0012) thickness
    Structural values are based on aluminum
        E = 70.e9            # [Pa] Young's modulus of the spar
        G = 30.e9            # [Pa] shear modulus of the spar
        stress = 20.e6       # [Pa] yield stress
        mrho = 3.e3          # [kg/m^3] material density
        fem_origin = 0.35    # chordwise location of the spar
    Other
        W0 = 0.4 * 3e5       # [kg] MTOW of B777 is 3e5 kg with fuel

    Default problem parameters:
    ---------------------------
    Re = 1e6                 # Reynolds number
    reynolds_length = 1.0    # characteristic Reynolds length
    alpha = 5.               # angle of attack
    CT = 9.80665 * 17.e-6    # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
    R = 14.3e6               # [m] maximum range
    M = 0.84                 # Mach number at cruise
    rho = 0.38               # [kg/m^3] air density at 35,000 ft
    a = 295.4                # [m/s] speed of sound at 35,000 ft
    with_viscous = False     # if true, compute viscous drag

    '''
    # Use steps in run_aerostruct.py to add wing surface to problem

    # Set problem type
    prob_dict.update({'type' : 'aerostruct'})  # this doesn't really matter since we aren't calling OASProblem.setup()

    # Instantiate problem
    OAS_prob = OASProblem(prob_dict)

    for surface in surfaces:
        # Add SpatialBeamFEM size
        FEMsize = 6 * surface['num_y'] + 6
        surface.update({'FEMsize': FEMsize})
        # Add the specified wing surface to the problem.
        OAS_prob.add_surface(surface)

    # Add materials properties for the wing surface to the surface dict in OAS_prob
    for idx, surface in enumerate(OAS_prob.surfaces):
        A, Iy, Iz, J = materials_tube(surface['r'], surface['t'], surface)
        OAS_prob.surfaces[idx].update({
            'A': A,
            'Iy': Iy,
            'Iz': Iz,
            'J': J
        })

    # Get total panels and save in prob_dict
    tot_panels = 0
    for surface in OAS_prob.surfaces:
        ny = surface['num_y']
        nx = surface['num_x']
        tot_panels += (nx - 1) * (ny - 1)
    OAS_prob.prob_dict.update({'tot_panels': tot_panels})

    # Assume we are only using a single lifting surface for now
    surface = OAS_prob.surfaces[0]

    # Initialize the OpenAeroStruct components and save them in a component dictionary
    comp_dict = {}
    comp_dict['MaterialsTube'] = MaterialsTube(surface)
    comp_dict['GeometryMesh'] = GeometryMesh(surface)
    comp_dict['TransferDisplacements'] = TransferDisplacements(surface)

    # Add bspline components for active bspline geometric variables
    comp_dict['Bspline'] = {}  # empty dict to contain all Bspline component sub dict
    for var in surface['active_bsp_vars']:
        n_pts = surface['num_y']
        if var == 'thickness_cp':
            n_pts -= 1
        trunc_var = var.split('_')[0]
        comp_dict['Bspline'].update({   # add dict containing initialized Bspline component for variable
            trunc_var: Bspline(var, trunc_var, surface['num_'+var], n_pts)
        })
    # incorporate bspline elements into surface mesh
    for trunc_var, bsp_comp in comp_dict['Bspline'].items():
        # trunc_var = truncated variable name, e.g. "thickness"
        # bsp_comp = Bspline() component for variable
        var = trunc_var + '_cp'
        n_pts = surface['num_y']
        if var == 'thickness_cp':
            n_pts -= 1
        surface[trunc_var] = b_spline(var, surface[var], n_pts, bsp_comp)
    surface['mesh'] = geometry_mesh(surface, comp_dict['GeometryMesh'])
    surface['disp'] = np.zeros((surface['num_y'], 6), dtype=data_type)  # zero displacement
    surface['def_mesh'] = transfer_displacements(surface['mesh'], surface['disp'], comp=comp_dict['TransferDisplacements'])
    # Initialize remaining OpenAeroStruct components
    comp_dict['VLMGeometry'] = VLMGeometry(surface)
    comp_dict['AssembleAIC'] = AssembleAIC([surface])
    comp_dict['AeroCirculations'] = AeroCirculations(OAS_prob.prob_dict['tot_panels'])
    comp_dict['VLMForces'] = VLMForces([surface])
    comp_dict['VLMLiftDrag'] = VLMLiftDrag(surface)
    comp_dict['VLMCoeffs'] = VLMCoeffs(surface)
    comp_dict['TotalLift'] = TotalLift(surface)
    comp_dict['TotalDrag'] = TotalDrag(surface)
    comp_dict['ViscousDrag'] = ViscousDrag(surface, OAS_prob.prob_dict['with_viscous'])
    comp_dict['TransferLoads'] = TransferLoads(surface)
    comp_dict['ComputeNodes'] = ComputeNodes(surface)
    comp_dict['AssembleK'] = AssembleK(surface)
    comp_dict['SpatialBeamFEM'] = SpatialBeamFEM(surface['FEMsize'])
    comp_dict['SpatialBeamDisp'] = SpatialBeamDisp(surface)
    comp_dict['SpatialBeamEnergy'] = SpatialBeamEnergy(surface)
    comp_dict['SpatialBeamWeight'] = SpatialBeamWeight(surface)
    comp_dict['SpatialBeamVonMisesTube'] = SpatialBeamVonMisesTube(surface)
    comp_dict['SpatialBeamFailureExact'] = SpatialBeamFailureExact(surface)
    comp_dict['SpatialBeamFailureKS'] = SpatialBeamFailureKS(surface)
    comp_dict['FunctionalBreguetRange'] = FunctionalBreguetRange([surface], OAS_prob.prob_dict)
    comp_dict['FunctionalEquilibrium'] = FunctionalEquilibrium([surface], OAS_prob.prob_dict)
    OAS_prob.comp_dict = comp_dict

    return OAS_prob


def b_spline(cpname, cpval, n_output, comp):
    ''' Add bspline components for active bspline geometric variables
        Note: keep n_input in as input argument for forward compatibility'''
    params = {
        cpname: cpval
    }
    unknowns = {
        cpname.split('_')[0]: np.zeros(n_output)
    }
    resids = {}
    comp.solve_nonlinear(params, unknowns, resids)
    return unknowns.get(comp.ptname)


def gen_init_mesh(surface, comp_dict=None):
    ''' Generate initial def_mesh '''
    if comp_dict:
        for trunc_var, bsp_comp in comp_dict['Bspline'].items():
            # trunc_var = truncated variable name, e.g. "thickness"
            # bsp_comp = Bspline() component for variable
            var = trunc_var + '_cp'
            n_pts = surface['num_y']
            if var == 'thickness_cp':
                n_pts -= 1
            surface[trunc_var] = b_spline(var, surface[var], n_pts, bsp_comp)
        mesh = geometry_mesh(surface, comp_dict['GeometryMesh'])
        disp = np.zeros((surface['num_y'], 6), dtype=data_type)  # zero displacement
        def_mesh = transfer_displacements(mesh, disp, comp=comp_dict['TransferDisplacements'])
        PRINT('@@@@@ mesh - def_mesh =',mesh-def_mesh)
    else:
        mesh = geometry_mesh(surface)
        disp = np.zeros((surface['num_y'], 6), dtype=data_type)  # zero displacement
        def_mesh = transfer_displacements(mesh, disp, surface)
    # update the surface dictionary
    surface['mesh'] = mesh
    surface['disp'] = disp
    surface['def_mesh'] = def_mesh
    return def_mesh


def aerodynamics(def_mesh, surface, prob_dict, comp_dict):
    ''' Use pre-initialized components '''

    # Unpack variables
    v = prob_dict.get('v')
    alpha = prob_dict.get('alpha')
    size = prob_dict.get('tot_panels')
    rho = prob_dict.get('rho')

    b_pts, c_pts, widths, cos_sweep, lengths, normals, S_ref = vlm_geometry(def_mesh, comp_dict['VLMGeometry'])
    AIC, rhs= assemble_aic(surface, def_mesh, b_pts, c_pts, normals, v, alpha, comp_dict['AssembleAIC'])
    circulations = aero_circulations(AIC, rhs, comp_dict['AeroCirculations'])
    sec_forces = vlm_forces(surface, def_mesh, b_pts, circulations, alpha, v, rho, comp_dict['VLMForces'])
    loads = transfer_loads(def_mesh, sec_forces, comp_dict['TransferLoads'])
    # store variables in surface dict
    surface.update({
        'b_pts': b_pts,
        'c_pts': c_pts,
        'widths': widths,
        'cos_sweep': cos_sweep,
        'lengths': lengths,
        'normals': normals,
        'S_ref': S_ref,
        'loads': loads,
        'sec_forces': sec_forces
    })
    prob_dict.update({
        'sec_forces': sec_forces
    })
    return loads


def aerodynamics2(def_mesh, surface, prob_dict):
    ''' Don't use pre-initialized components '''

    # Unpack variables
    v = prob_dict.get('v')
    alpha = prob_dict.get('alpha')
    size = prob_dict.get('tot_panels')
    rho = prob_dict.get('rho')

    b_pts, c_pts, widths, cos_sweep, lengths, normals, S_ref = vlm_geometry(def_mesh, surface)
    AIC, rhs= assemble_aic(surface, def_mesh, b_pts, c_pts, normals, v, alpha)
    circulations = aero_circulations(AIC, rhs, size)
    sec_forces = vlm_forces(surface, def_mesh, b_pts, circulations, alpha, v, rho)
    loads = transfer_loads(def_mesh, sec_forces, surface)

    return loads


def structures(loads, surface, prob_dict, comp_dict):
    ''' Use pre-initialized components '''

    # Unpack variables
    A = surface.get('A')
    Iy = surface.get('Iy')
    Iz = surface.get('Iz')
    J = surface.get('J')
    mesh = surface.get('mesh')
    v = prob_dict.get('v')
    alpha = prob_dict.get('alpha')
    size =  prob_dict.get('tot_panels')

    nodes = compute_nodes(mesh, comp_dict['ComputeNodes'])
    K, forces = assemble_k(A, Iy, Iz, J, nodes, loads, comp_dict['AssembleK'])
    disp_aug = spatial_beam_fem(K, forces, comp_dict['SpatialBeamFEM'])
    disp = spatial_beam_disp(disp_aug, comp_dict['SpatialBeamDisp'])
    def_mesh = transfer_displacements(mesh, disp, comp_dict['TransferDisplacements'])
    surface.update({
        'K': K,
        'forces': forces,
        'nodes': nodes,
        'def_mesh': def_mesh,
        'disp_aug': disp_aug,
        'disp': disp
    })
    return def_mesh  # Output the def_mesh matrix


def structures2(loads, surface, prob_dict):
    ''' Don't use pre-initialized components '''

    # Unpack variables
    A = surface.get('A')
    Iy = surface.get('Iy')
    Iz = surface.get('Iz')
    J = surface.get('J')
    mesh = surface.get('mesh')
    FEMsize =  surface.get('FEMsize')
    v = prob_dict.get('v')
    alpha = prob_dict.get('alpha')
     # Add the specified wing surface to the problem.

    nodes = compute_nodes(mesh, surface)
    K, forces = assemble_k(A, Iy, Iz, J, nodes, loads, surface)
    disp_aug = spatial_beam_fem(K, forces, FEMsize)
    disp = spatial_beam_disp(disp_aug, surface)
    def_mesh = transfer_displacements(mesh, disp, surface)

    return def_mesh  # Output the def_mesh matrix


def aero_perf(surface, prob_dict, comp_dict):
    # unpack surface variables
    S_ref = surface.get('S_ref')
    cos_sweep = surface.get('cos_sweep')
    widths = surface.get('widths')
    lengths = surface.get('lengths')
    sec_forces = surface.get('sec_forces')
    # unpack problem variables
    M = prob_dict.get('M')
    re = prob_dict.get('re')
    rho = prob_dict.get('rho')
    alpha = prob_dict.get('alpha')
    v = prob_dict.get('v')

    CDv = viscous_drag(re, M, S_ref, cos_sweep, widths, lengths, comp_dict['ViscousDrag'])
    L, D = vlm_lift_drag(sec_forces, alpha, comp_dict['VLMLiftDrag'])
    CL1, CDi = vlm_coeffs(S_ref, L, D, v, rho, comp_dict['VLMCoeffs'])
    CL = total_lift(CL1, comp_dict['TotalLift'])
    CD = total_drag(CDi, CDv, comp_dict['TotalDrag'])
    # store surface variables in dict
    surface.update({
        'CDv': CDv,
        'L': L,
        'D': D,
        'CL1': CL1,
        'CDi': CDi,
        'CL': CL,
        'CD': CD
    })
    return


def struct_perf(surface, prob_dict, comp_dict):
    # unpack surface variables
    disp = surface.get('disp')
    loads = surface.get('loads')
    nodes = surface.get('nodes')
    r = surface.get('r')
    A = surface.get('A')
    energy = spatialbeam_energy(disp, loads, comp_dict['SpatialBeamEnergy'])
    weight = spatialbeam_weight(A, nodes, comp_dict['SpatialBeamWeight'])
    vonmises = spatialbeam_vonmises_tube(r, nodes, disp, comp_dict['SpatialBeamVonMisesTube'])
    if surface['exact_failure_constraint']:
        failure = spatialbeam_failure_exact(vonmises, comp_dict['SpatialBeamFailureExact'])
    else:
        failure = spatialbeam_failure_ks(vonmises, comp_dict['SpatialBeamFailureKS'])
    # save surface variables
    surface.update({
        'energy': energy,
        'weight': weight,
        'vonmises': vonmises,
        'failure': failure
    })
    return


def geometry_mesh(surface, comp=None):
    """
    OpenMDAO component that performs mesh manipulation functions. It reads in
    the initial mesh from the surface dictionary and outputs the altered
    mesh based on the geometric design variables.

    Parameters
    ----------
    sweep : float
        Shearing sweep angle in degrees.
    dihedral : float
        Dihedral angle in degrees.
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.
    chord_dist[ny] : numpy array
        Chord length for each panel edge.
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.
    comp : (optional) OpenAeroStruct component object.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface dictionary and
        the geometric design variables.
    """
    if not comp:
        comp = GeometryMesh(surface)
    params = {}
    #
    # The following is copied from the __init__() method of GeometryMesh()
    #
    ny = surface['num_y']
    ones_list = ['taper', 'chord_cp']     # Variables that should be initialized to one
    zeros_list = ['sweep', 'dihedral', 'twist_cp', 'xshear_cp', 'zshear_cp']     # Variables that should be initialized to zero
    set_list = ['span']     # Variables that should be initialized to given value
    all_geo_vars = ones_list + zeros_list + set_list
    for var in all_geo_vars:
        if len(var.split('_')) > 1:
            param = var.split('_')[0]
        else:
            param = var
        if var in surface['active_geo_vars']:
            params.update({param: surface[param]})
    unknowns = {
        'mesh': comp.mesh
    }
    print(unknowns)
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    mesh = unknowns.get('mesh')
    return mesh


def transfer_displacements(mesh, disp, comp):
    """
    Perform displacement transfer.

    Apply the computed displacements on the original mesh to obtain
    the deformed mesh.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces.
    disp[ny, 6] : numpy array
        Flattened array containing displacements on the FEM component.
        Contains displacements for all six degrees of freedom, including
        displacements in the x, y, and z directions, and rotations about the
        x, y, and z axes.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp = TransferDisplacements(surface)
    params = {
        'mesh': mesh,
        'disp': disp
    }
    unknowns = {
        'def_mesh': np.zeros((comp.nx, comp.ny, 3), dtype=data_type)
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    def_mesh = unknowns.get('def_mesh')
    return def_mesh


"""
================================================================================
                                AERODYNAMICS
================================================================================
From vlm.py: """


def vlm_geometry(def_mesh, comp):
    """ Compute various geometric properties for VLM analysis.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    widths[nx-1, ny-1] : numpy array
        The spanwise widths of each individual panel.
    lengths[ny] : numpy array
        The chordwise length of the entire airfoil following the camber line.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    S_ref : float
        The reference area of the lifting surface.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp = VLMGeometry(surface)
    params = {
        'def_mesh': def_mesh
    }
    unknowns = {
        'b_pts': np.zeros((comp.nx-1, comp.ny, 3), dtype=data_type),
        'c_pts': np.zeros((comp.nx-1, comp.ny-1, 3)),
        'widths': np.zeros((comp.ny-1)),
        'cos_sweep': np.zeros((comp.ny-1)),
        'lengths': np.zeros((comp.ny)),
        'normals': np.zeros((comp.nx-1, comp.ny-1, 3)),
        'S_ref': 0.
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    b_pts=unknowns.get('b_pts')
    c_pts=unknowns.get('c_pts')
    widths=unknowns.get('widths')
    cos_sweep=unknowns.get('cos_sweep')
    lengths=unknowns.get('lengths')
    normals=unknowns.get('normals')
    S_ref=unknowns.get('S_ref')
    return b_pts, c_pts, widths, cos_sweep, lengths, normals, S_ref


def assemble_aic(surface, def_mesh, b_pts, c_pts, normals, v, alpha, comp=None):
    """
    Compute the circulations based on the AIC matrix and the panel velocities.
    Note that the flow tangency condition is enforced at the 3/4 chord point.
    There are multiple versions of the first four parameters with one
    for each surface defined.
    Each of these parameters has the name of the surface prepended on the
    actual parameter name.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.

    v : float
        Freestream air velocity in m/s.
    alpha : float
        Angle of attack in degrees.
    comp : (Optional) OpenAeroStruct component object.

    Returns
    -------
    AIC[(nx-1)*(ny-1), (nx-1)*(ny-1)] : numpy array
        The aerodynamic influence coefficient matrix. Solving the linear system
        of AIC * circulations = n * v gives us the circulations for each of the
        horseshoe vortices.
    rhs[(nx-1)*(ny-1)] : numpy array
        The right-hand-side of the linear system that yields the circulations.
    """
    surfaces = [surface]
    if not comp:
        comp=AssembleAIC(surfaces)
    params = {}
    ny=surface['num_y']
    nx=surface['num_x']
    name=surface['name']
    params.update({
        name + 'def_mesh': def_mesh,
        name + 'b_pts': b_pts,
        name + 'c_pts': c_pts,
        name + 'normals': normals
    })
    params.update({
        'v': v,
        'alpha': alpha
    })
    unknowns={
        'AIC': np.zeros((comp.tot_panels, comp.tot_panels), dtype = data_type),
        'rhs': np.zeros((comp.tot_panels), dtype = data_type)
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    AIC=unknowns.get('AIC')
    rhs=unknowns.get('rhs')
    return AIC, rhs


def aero_circulations(AIC, rhs, comp):
    """
    Compute the circulation strengths of the horseshoe vortices by solving the
    linear system AIC * circulations = n * v.
    This component is copied from OpenMDAO's LinearSystem component with the
    names of the parameters and outputs changed to match our problem formulation.

    Parameters
    ----------
    AIC[(nx-1)*(ny-1), (nx-1)*(ny-1)] : numpy array
        The aerodynamic influence coefficient matrix. Solving the linear system
        of AIC * circulations = n * v gives us the circulations for each of the
        horseshoe vortices.
    rhs[(nx-1)*(ny-1)] : numpy array
        The right-hand-side of the linear system that yields the circulations.
    comp : Either OpenAeroStruct component object (better), or tot_panels.


    Returns
    -------
    circulations[(nx-1)*(ny-1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        AIC * circulations = n * v.
    """
    if not isinstance(comp, Component):
        tot_panels = comp
        comp = AeroCirculations(tot_panels)
    size = comp.size
    params = {
        'AIC': AIC,
        'rhs': rhs
    }
    unknowns = {
        'circulations': np.zeros((size), dtype=data_type)
    }
    resids = {
        'circulations': np.zeros((size), dtype=data_type)
    }
    comp.solve_nonlinear(params, unknowns, resids)
    circulations = unknowns.get('circulations')
    return circulations


def vlm_forces(surface, def_mesh, b_pts, circulations, alpha, v, rho, comp=None):
    """ Compute aerodynamic forces acting on each section.

    Note that the first two parameters and the unknown have the surface name
    prepended on it. E.g., 'def_mesh' on a surface called 'wing' would be
    'wing.def_mesh', etc.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.

    circulations : numpy array
        Flattened vector of horseshoe vortex strengths calculated by solving
        the linear system of AIC_mtx * circulations = rhs, where rhs is
        based on the air velocity at each collocation point.
    alpha : float
        Angle of attack in degrees.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.
    comp : (optional) OpenAeroStruct component object.

    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant with more than one chordwise
        panel).

    """
    surfaces = [surface]
    if not comp:
        comp=VLMForces(surfaces)
    params = {}
    unknowns = {}
    tot_panels = 0
    name = surface['name']
    ny = surface['num_y']
    nx = surface['num_x']
    tot_panels += (nx - 1) * (ny - 1)
    params.update({
        name+'def_mesh': def_mesh,
        name+'b_pts': b_pts
    })
    unknowns.update({
        name+'sec_forces': np.zeros((nx-1, ny-1, 3), dtype=data_type)
    })
    params.update({
        'circulations': circulations,
        'alpha': alpha,
        'v': v,
        'rho': rho
    })
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    sec_forces=unknowns.get(name+'sec_forces')
    return sec_forces


def transfer_loads(def_mesh, sec_forces, comp):
    """
    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp=TransferLoads(surface)
    params={
        'def_mesh': def_mesh,
        'sec_forces': sec_forces
    }
    unknowns={
        'loads': np.zeros((comp.ny, 6), dtype=data_type)
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    loads=unknowns.get('loads')
    return loads


def vlm_lift_drag(sec_forces, alpha, comp):
    """
    Calculate total lift and drag in force units based on section forces.

    Parameters
    ----------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant with more than one chordwise
        panel).
    alpha : float
        Angle of attack in degrees.

    Returns
    -------
    L : float
        Total induced lift force for the lifting surface.
    D : float
        Total induced drag force for the lifting surface.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp=VLMLiftDrag(surface)
    params={
        'sec_forces': sec_forces,
        'alpha': alpha
    }
    unknowns={
        'L': 0.,
        'D': 0.
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    L=unknowns.get('L')
    D=unknowns.get('D')
    return L, D


def vlm_coeffs(S_ref, L, D, v, rho, comp):
    """ Compute lift and drag coefficients.

    Parameters
    ----------
    S_ref : float
        The reference areas of the lifting surface.
    L : float
        Total lift for the lifting surface.
    D : float
        Total drag for the lifting surface.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp = VLMCoeffs(surface)
    params = {
        'S_ref': S_ref,
        'L': L,
        'D': D,
        'v': v,
        'rho': rho
    }
    unknowns = {
        'CL1': 0.,
        'CDi': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    CL1 = unknowns.get('CL1', 0.)
    CDi = unknowns.get('CDi', 0.)
    return CL1, CDi


def total_lift(CL1, comp):
    """ Calculate total lift in force units.

    Parameters
    ----------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.

    Returns
    -------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp = TotalLift(surface)
    params = {
        'CL1': CL1
    }
    unknowns = {
        'CL': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    CL = unknowns.get('CL', 0.)
    return CL


def total_drag(CDi, CDv, comp):
    """ Calculate total drag in force units.

    Parameters
    ----------
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    CDv : float
        Calculated viscous drag for the lifting surface..

    Returns
    -------
    CD : float
        Total coefficient of drag (CD) for the lifting surface.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp = TotalDrag(surface)
    params = {
        'CDi': CDi,
        'CDv': CDv
    }
    unknowns = {
        'CD': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    CD = unknowns.get('CD', 0.)
    return CD


def viscous_drag(Re, M, S_ref, cos_sweep, widths, lengths, comp, withViscous=True):
    """
    Compute the skin friction drag if the with_viscous option is True.

    Parameters
    ----------
    re : float
        Dimensionalized (1/length) Reynolds number. This is used to compute the
        local Reynolds number based on the local chord length.
    M : float
        Mach number.
    S_ref : float
        The reference area of the lifting surface.
    sweep : float
        The angle (in degrees) of the wing sweep. This is used in the form
        factor calculation.
    widths[ny-1] : numpy array
        The spanwise width of each panel.
    lengths[ny] : numpy array
        The sum of the lengths of each line segment along a chord section.

    Returns
    -------
    CDv : float
        Viscous drag coefficient for the lifting surface computed using flat
        plate skin friction coefficient and a form factor to account for wing
        shape.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp=ViscousDrag(surface, withViscous)
    params = {
        're': Re,
        'M': M,
        'S_ref': S_ref,
        'cos_sweep': cos_sweep,
        'widths': widths,
        'lengths': lengths
    }
    unknowns = {
        'CDv': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    CDv = unknowns.get('CDv',0.0)
    return CDv


"""
================================================================================
                                   STRUCTURES
================================================================================
From spatialbeam.py: Define the structural analysis component using spatial beam theory. """

def spatial_beam_fem(K, forces, comp):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.
    This component is copied from OpenMDAO's LinearSystem component with the
    names of the parameters and outputs changed to match our problem formulation.

    Parameters
    ----------
    K[6*(ny+1), 6*(ny+1)] : numpy array
        Stiffness matrix for the entire FEM system. Used to solve the linear
        system K * u = f to obtain the displacements, u.
    forces[6*(ny+1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.
    comp : Either OpenAeroStruct component object (better), or FEMsize of surface.


    Returns
    -------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        K * u = f, where f is a flattened version of loads.

    """
    if not isinstance(comp, Component):
        FEMsize = comp
        comp=SpatialBeamFEM(FEMsize)
    else:
        FEMsize = comp.size
    params={
        'K': K,
        'forces': forces
    }
    unknowns={
        'disp_aug': np.zeros((FEMsize), dtype=data_type)
    }
    resids={
        'disp_aug': np.zeros((FEMsize), dtype=data_type)
    }
    comp.solve_nonlinear(params, unknowns, resids)
    disp_aug=unknowns.get('disp_aug')
    return disp_aug


def spatial_beam_disp(disp_aug, comp):
    """
    Reshape the flattened displacements from the linear system solution into
    a 2D array so we can more easily use the results.

    The solution to the linear system has additional results due to the
    constraints on the FEM model. The displacements from this portion of
    the linear system are not needed, so we select only the relevant
    portion of the displacements for further calculations.

    Parameters
    ----------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = forces, where forces is a flattened version of loads.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    disp[6*ny] : numpy array
        Actual displacement array formed by truncating disp_aug.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp=SpatialBeamDisp(surface)
    params={
        'disp_aug': disp_aug
    }
    unknowns={
        'disp': np.zeros((comp.ny, 6), dtype=data_type)
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    disp=unknowns.get('disp')
    return disp


def compute_nodes(mesh, comp):
    """
    Compute FEM nodes based on aerodynamic mesh.

    The FEM nodes are placed at fem_origin * chord,
    with the default fem_origin = 0.35.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp=ComputeNodes(surface)
    params={
        'mesh': mesh
    }
    unknowns={
        'nodes': np.zeros((comp.ny, 3), dtype=data_type)
    }
    resids=None
    comp.solve_nonlinear(params, unknowns, resids)
    nodes=unknowns.get('nodes')
    return nodes


def assemble_k(A, Iy, Iz, J, nodes, loads, comp):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.

    Parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    Iy[ny-1] : numpy array
        Mass moment of inertia around the y-axis for each FEM element.
    Iz[ny-1] : numpy array
        Mass moment of inertia around the z-axis for each FEM element.
    J[ny-1] : numpy array
        Polar moment of inertia for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    K[(nx-1)*(ny-1), (nx-1)*(ny-1)] : numpy array
        Stiffness matrix for the entire FEM system. Used to solve the linear
        system K * u = f to obtain the displacements, u.
    forces[(nx-1)*(ny-1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp = AssembleK(surface)  # if component is not passed in, surface must be
    params = {
        'A': A,
        'Iy': Iy,
        'Iz': Iz,
        'J': J,
        'nodes': nodes,
        'loads': loads
    }
    unknowns = {
        'K': np.zeros((comp.size, comp.size), dtype=data_type),
        'forces': np.zeros((comp.size), dtype=data_type)
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    K = unknowns.get('K')
    forces = unknowns.get('forces')
    return K, forces


def spatialbeam_energy(disp, loads, comp):
    """ Compute strain energy.

    Parameters
    ----------
    disp[ny, 6] : numpy array
        Actual displacement array formed by truncating disp_aug.
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    energy : float
        Total strain energy of the structural component.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp = SpatialBeamEnergy(surface)
    params = {
        'disp': disp,
        'loads': loads
    }
    unknowns = {
        'energy': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    energy = unknowns.get('energy', 0.)
    return energy


def spatialbeam_weight(A, nodes, comp):
    """ Compute total weight.

    Parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    weight : float
        Total weight of the structural component.
    """
    if not isinstance(comp, Component):
        surface = comp
        comp = SpatialBeamWeight(surface)
    params = {
        'A': A,
        'nodes': nodes
    }
    unknowns = {
        'weight': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    weight = unknowns.get('weight', 0.)
    return weight

def spatialbeam_vonmises_tube(r, nodes, disp, comp):
    """ Compute the von Mises stress in each element.

    Parameters
    ----------
    r[ny-1] : numpy array
        Radii for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    disp[ny, 6] : numpy array
        Displacements of each FEM node.

    Returns
    -------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp = SpatialBeamVonMisesTube(surface)
    params = {
        'nodes': nodes,
        'r': r,
        'disp': disp
    }
    unknowns = {
        'vonmises': np.zeros((comp.ny-1, 2), dtype=data_type)
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    vonmises = unknowns.get('vonmises')
    return vonmises


def spatialbeam_failure_ks(vonmises, comp):
    """
    Aggregate failure constraints from the structure.

    To simplify the optimization problem, we aggregate the individual
    elemental failure constraints using a Kreisselmeier-Steinhauser (KS)
    function.

    The KS function produces a smoother constraint than using a max() function
    to find the maximum point of failure, which produces a better-posed
    optimization problem.

    The rho parameter controls how conservatively the KS function aggregates
    the failure constraints. A lower value is more conservative while a greater
    value is more aggressive (closer approximation to the max() function).

    Parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp = SpatialBeamFailureKS(surface)
    params = {
        'vonmises': vonmises
    }
    unknowns = {
        'failure': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    failure = unknowns.get('failure')
    return failure


def spatialbeam_failure_exact(vonmises, comp):
    """
    Outputs individual failure constraints on each FEM element.

    Parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure[ny-1, 2] : numpy array
        Array of failure conditions. Positive if element has failed.

    """
    if not isinstace(comp, Component):
        surface = comp
        comp = SpatialBeamFailureExact(surface)
    params = {
        'vonmises': vonmises
    }
    unknowns = {
        'failure': np.zeros((comp.ny-1, 2), dtype=data_type)
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    failure = unknowns.get('failure')
    return failure


"""
================================================================================
                                MATERIALS
================================================================================
From materials.py: """


def materials_tube(r, thickness, comp):
    """ Compute geometric properties for a tube element.

    Parameters
    ----------
    r : array_like
        Radii for each FEM element.
    thickness : array_like
        Tube thickness for each FEM element.
    comp : Either OpenAeroStruct component object (better), or surface dict.

    Returns
    -------
    A : array_like
        Areas for each FEM element.
    Iy : array_like
        Mass moment of inertia around the y-axis for each FEM element.
    Iz : array_like
        Mass moment of inertia around the z-axis for each FEM element.
    J : array_like
        Polar moment of inertia for each FEM element.

    """
    if not isinstance(comp, Component):
        surface = comp
        comp=MaterialsTube(surface)
    # if not r:
    #     r = surface['r']  # this is already contained in surface dict
    # if not thickness:
    #     thickness = surface['t']  # this is already contained in surface dict
    params={
        'r': r,
        'thickness': thickness
    }
    unknowns={
        'A': np.zeros((comp.ny - 1)),
        'Iy': np.zeros((comp.ny - 1)),
        'Iz': np.zeros((comp.ny - 1)),
        'J': np.zeros((comp.ny - 1))
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    A=unknowns.get('A')
    Iy=unknowns.get('Iy')
    Iz=unknowns.get('Iz')
    J=unknowns.get('J')
    return A, Iy, Iz, J

    """
================================================================================
                                FUNCTIONALS
================================================================================
    From functionals.py:
    """

def functional_breguet_range(surfaces, CL, CD, weight, prob_dict, comp):
    """ Computes the fuel burn using the Breguet range equation """
    if not isinstance(comp, Component):
        surfaces = comp
        comp = FunctionalBreguetRange(surfaces, prob_dict)
    params = {}
    for surface in surfaces:
        name = surface['name']
        params.update({
            name+'CL': CL,
            name+'CD': CD,
            name+'weight': weight
        })
    unknowns = {
        'fuelburn': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    fuelburn = unknowns.get('fuelburn', 0.)
    return fuelburn


def functional_equilibrium(surfaces, L, weight, fuelburn, prob_dict, comp):
    """ L = W constraint """
    if not isinstance(comp, Component):
        surfaces = comp
        comp = FunctionalEquilibrium(surfaces, prob_dict)
    params = {}
    for surface in surfaces:
        name = surface['name']
        params.update({
            name+'L': L,
            name+'weight': weight
        })
    params.update({
        'fuelburn': fuelburn
    })
    unknowns = {
        'eq_con': 0.
    }
    resids = None
    comp.solve_nonlinear(params, unknowns, resids)
    eq_con = unknowns.get('eq_con', 0.)
    return eq_con


def aerodymanics3(def_mesh, AeroProb):
    AeroProb['def_mesh'] = def_mesh
    AeroProb.run()
    return AeroProb['loads']

def structures3(loads, StructProb):
    StructProb['loads'] = loads
    StructProb.run()
    return StructProb['def_mesh']

def setup_AeroProb(prob_dict={}, surfaces=[{}]):
    """
    Specific method to add the necessary components to the problem for an
    aerodynamic problem.
    """

    # Set problem type
    prob_dict.update({'type' : 'aero'})

    # Instantiate problem
    AeroProb = Problem()

    for surface in surfaces:
        # Add SpatialBeamFEM size
        FEMsize = 6 * surface['num_y'] + 6
        surface.update({'FEMsize': FEMsize})
        # Add the specified wing surface to the problem.
        OAS_prob.add_surface(surface)

    # Add materials properties for the wing surface to the surface dict in OAS_prob
    for idx, surface in enumerate(OAS_prob.surfaces):
        A, Iy, Iz, J = materials_tube(surface['r'], surface['t'], surface)
        OAS_prob.surfaces[idx].update({
            'A': A,
            'Iy': Iy,
            'Iz': Iz,
            'J': J
        })

    # Get total panels and save in prob_dict
    tot_panels = 0
    for surface in OAS_prob.surfaces:
        ny = surface['num_y']
        nx = surface['num_x']
        tot_panels += (nx - 1) * (ny - 1)
    OAS_prob.prob_dict.update({'tot_panels': tot_panels})

    # Assume we are only using a single lifting surface for now
    surface = OAS_prob.surfaces[0]

    # Set the problem name if the user doesn't
    if 'prob_name' not in self.prob_dict.keys():
        self.prob_dict['prob_name'] = 'aero'

    # Create the base root-level group
    root = Group()

    # Create the problem and assign the root group
    self.prob = Problem()
    self.prob.root = root

    # Loop over each surface in the surfaces list
    for surface in self.surfaces:

        # Get the surface name and create a group to contain components
        # only for this surface
        name = surface['name']
        tmp_group = Group()

        # Add independent variables that do not belong to a specific component
        # indep_vars = [('disp', np.zeros((surface['num_y'], 6), dtype=data_type))]
        indep_vars = [('def_mesh', np.zeros((surface['num_y'], 6), dtype=data_type))]

        for var in surface['active_geo_vars']:
            indep_vars.append((var, surface[var]))

        # Add aero components to the surface-specific group
        tmp_group.add('indep_vars',
                 IndepVarComp(indep_vars),
                 promotes=['*'])
        tmp_group.add('mesh',
                 GeometryMesh(surface),
                 promotes=['*'])
        tmp_group.add('def_mesh',
                 TransferDisplacements(surface),
                 promotes=['*'])
        tmp_group.add('vlmgeom',
                 VLMGeometry(surface),
                 promotes=['*'])
        # Add bspline components for active bspline geometric variables
        for var in surface['active_bsp_vars']:
            n_pts = surface['num_y']
            if var == 'thickness_cp':
                n_pts -= 1
            trunc_var = var.split('_')[0]
            tmp_group.add(trunc_var + '_bsp',
                     Bspline(var, trunc_var, surface['num_'+var], n_pts),
                     promotes=['*'])
        if surface['monotonic_con'] is not None:
            if type(surface['monotonic_con']) is not list:
                surface['monotonic_con'] = [surface['monotonic_con']]
            for var in surface['monotonic_con']:
                tmp_group.add('monotonic_' + var,
                    MonotonicConstraint(var, surface), promotes=['*'])

# -----------------------------------------------------------

        # Add tmp_group to the problem as the name of the surface.
        # Note that is a group and performance group for each
        # individual surface.
        name_orig = name.strip('_')
        root.add(name_orig, tmp_group, promotes=[])
        root.add(name_orig+'_perf', VLMFunctionals(surface, self.prob_dict),
                promotes=["v", "alpha", "M", "re", "rho"])

    # Add problem information as an independent variables component
    if self.prob_dict['Re'] == 0:
        Error('Reynolds number must be greater than zero for viscous drag ' +
        'calculation. If only inviscid drag is desired, set with_viscous ' +
        'flag to False.')

    prob_vars = [('v', self.prob_dict['v']),
        ('alpha', self.prob_dict['alpha']),
        ('M', self.prob_dict['M']),
        ('re', self.prob_dict['Re']/self.prob_dict['reynolds_length']),
        ('rho', self.prob_dict['rho'])]
    root.add('prob_vars',
             IndepVarComp(prob_vars),
             promotes=['*'])

    # Add a single 'aero_states' component that solves for the circulations
    # and forces from all the surfaces.
    # While other components only depends on a single surface,
    # this component requires information from all surfaces because
    # each surface interacts with the others.
    root.add('aero_states',
             VLMStates(self.surfaces),
             promotes=['circulations', 'v', 'alpha', 'rho'])

    # Explicitly connect parameters from each surface's group and the common
    # 'aero_states' group.
    # This is necessary because the VLMStates component requires information
    # from each surface, but this information is stored within each
    # surface's group.
    for surface in self.surfaces:
        name = surface['name']

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        root.connect(name[:-1] + '.def_mesh', 'aero_states.' + name + 'def_mesh')
        root.connect(name[:-1] + '.b_pts', 'aero_states.' + name + 'b_pts')
        root.connect(name[:-1] + '.c_pts', 'aero_states.' + name + 'c_pts')
        root.connect(name[:-1] + '.normals', 'aero_states.' + name + 'normals')

        # Connect the results from 'aero_states' to the performance groups
        root.connect('aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

        # Connect S_ref for performance calcs
        root.connect(name[:-1] + '.S_ref', name + 'perf' + '.S_ref')
        root.connect(name[:-1] + '.widths', name + 'perf' + '.widths')
        root.connect(name[:-1] + '.lengths', name + 'perf' + '.lengths')
        root.connect(name[:-1] + '.cos_sweep', name + 'perf' + '.cos_sweep')

    # Actually set up the problem
    self.setup_prob()

class OASCoupledProblem(OASProblem):
    def __init__(self, input_dict={}):
        super(OASCoupledProblem, self).__init__(input_dict)
        # New coupled analysis setup functions
        if self.prob_dict['type'] = 'coupledsetup':
            self.setup = self.setup_coupledsetup
        if self.prob_dict['type'] = 'coupledaero':
            self.setup = self.setup_coupledaero
        if self.prob_dict['type'] = 'coupledstruct':
            self.setup = self.setup_coupledstruct

    def self.setup_coupledsetup(self):
        """
        Specific method to add the necessary components to the problem for a
        structural problem.
        """

        # Set the problem name if the user doesn't
        if 'prob_name' not in self.prob_dict.keys():
            self.prob_dict['prob_name'] = 'struct'

        # Create the base root-level group
        root = Group()

        # Create the problem and assign the root group
        self.prob = Problem()
        self.prob.root = root

        # Loop over each surface in the surfaces list
        for surface in self.surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface.
            # This group's name is whatever the surface's name is.
            # The default is 'wing'.
            name = surface['name']
            tmp_group = Group()

            # Add independent variables that do not belong to a specific component.
            # Note that these are the only ones necessary for structual-only
            # analysis and optimization.
            indep_vars = [('r', surface['r']), ('loads', surface['loads'])]
            for var in surface['active_geo_vars']:
                indep_vars.append((var, surface[var]))

            # Add structural components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('tube',
                     MaterialsTube(surface),
                     promotes=['*'])

            # Add bspline components for active bspline geometric variables
            for var in surface['active_bsp_vars']:
                n_pts = surface['num_y']
                if var == 'thickness_cp':
                    n_pts -= 1
                trunc_var = var.split('_')[0]
                tmp_group.add(trunc_var + '_bsp',
                         Bspline(var, trunc_var, surface['num_'+var], n_pts),
                         promotes=['*'])

            # Add tmp_group to the problem with the name of the surface.
            # The default is 'wing'.
            root.add(name[:-1], tmp_group, promotes=[])

        # Actually set up the problem
        self.setup_prob()

    def self.setup_coupledaero(self):
        """
        Specific method to add the necessary components to the problem for an
        coupled-only aerodynamic problem.
        """
        # Set the problem name if the user doesn't
        if 'prob_name' not in self.prob_dict.keys():
            self.prob_dict['prob_name'] = 'coupledaero'

        # Create the base root-level group
        root = Group()

        # Create the problem and assign the root group
        self.prob = Problem()
        self.prob.root = root

        # Loop over each surface in the surfaces list
        for surface in self.surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface['name']
            tmp_group = Group()

            # Add independent variables that do not belong to a specific component
            indep_vars = [('disp', np.zeros((surface['num_y'], 6), dtype=data_type))]
            for var in surface['active_geo_vars']:
                indep_vars.append((var, surface[var]))

            # Add aero components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('def_mesh',
                     TransferDisplacements(surface),
                     promotes=['*'])
            tmp_group.add('vlmgeom',
                     VLMGeometry(surface),
                     promotes=['*'])
            # Add bspline components for active bspline geometric variables
            for var in surface['active_bsp_vars']:
                n_pts = surface['num_y']
                if var == 'thickness_cp':
                    n_pts -= 1
                trunc_var = var.split('_')[0]
                tmp_group.add(trunc_var + '_bsp',
                         Bspline(var, trunc_var, surface['num_'+var], n_pts),
                         promotes=['*'])
            if surface['monotonic_con'] is not None:
                if type(surface['monotonic_con']) is not list:
                    surface['monotonic_con'] = [surface['monotonic_con']]
                for var in surface['monotonic_con']:
                    tmp_group.add('monotonic_' + var,
                        MonotonicConstraint(var, surface), promotes=['*'])

            # Add tmp_group to the problem as the name of the surface.
            # Note that is a group and performance group for each
            # individual surface.
            name_orig = name.strip('_')
            root.add(name_orig, tmp_group, promotes=[])
            root.add(name_orig+'_perf', VLMFunctionals(surface, self.prob_dict),
                    promotes=["v", "alpha", "M", "re", "rho"])

        # Add problem information as an independent variables component
        if self.prob_dict['Re'] == 0:
            Error('Reynolds number must be greater than zero for viscous drag ' +
            'calculation. If only inviscid drag is desired, set with_viscous ' +
            'flag to False.')

        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('re', self.prob_dict['Re']/self.prob_dict['reynolds_length']),
            ('rho', self.prob_dict['rho'])]
        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        # Add a single 'aero_states' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        root.add('aero_states',
                 VLMStates(self.surfaces),
                 promotes=['circulations', 'v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        # This is necessary because the VLMStates component requires information
        # from each surface, but this information is stored within each
        # surface's group.
        for surface in self.surfaces:
            name = surface['name']

            # Perform the connections with the modified names within the
            # 'aero_states' group.
            root.connect(name[:-1] + '.def_mesh', 'aero_states.' + name + 'def_mesh')
            root.connect(name[:-1] + '.b_pts', 'aero_states.' + name + 'b_pts')
            root.connect(name[:-1] + '.c_pts', 'aero_states.' + name + 'c_pts')
            root.connect(name[:-1] + '.normals', 'aero_states.' + name + 'normals')

            # Connect the results from 'aero_states' to the performance groups
            root.connect('aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

            # Connect S_ref for performance calcs
            root.connect(name[:-1] + '.S_ref', name + 'perf' + '.S_ref')
            root.connect(name[:-1] + '.widths', name + 'perf' + '.widths')
            root.connect(name[:-1] + '.lengths', name + 'perf' + '.lengths')
            root.connect(name[:-1] + '.cos_sweep', name + 'perf' + '.cos_sweep')

        # Actually set up the problem
        self.setup_prob()

    def self.setup_coupledstruct(self):
        pass

if __name__ == "__main__":
    ''' Test the coupled system with default parameters

     To change problem parameters, input the prob_dict dictionary, e.g.
     prob_dict = {
        'rho' : 0.35,
        'R': 14.0e6
     }
    '''
    print('Fortran Flag = {0}'.format(fortran_flag))

    # Define parameters
    prob_dict = {'type':'aerostruct'} # use default

    # Define surface
    surface = {
        'name': 'wing',
        'wing_type' : 'CRM',
        'num_x': 2,   # number of chordwise points
        'num_y': 9    # number of spanwise points
    }

    # Define fixed point iteration options
    # default options from OpenMDAO nonlinear solver NLGaussSeidel
    fpi_opt = {
        # 'atol': float(1e-06),   # Absolute convergence tolerance (unused)
        # 'err_on_maxiter': bool(False),  # raise AnalysisError if not converged at maxiter (unused)
        # 'print': int(0),        # Print option (unused)
        'maxiter': int(100),    # Maximum number of iterations
        # 'rtol': float(1e-06),   # Relative convergence tolerance (unused)
        'utol': float(1e-16)    # Convergence tolerance on the change in the unknowns
    }

    print('Run analysis.setup()')

    OASprob = setup(prob_dict, [surface])

    # Using standard method in run_classes.py
    stdOASprob = OASProblem(prob_dict)
    stdOASprob.add_surface(surface)
    stdOASprob.setup()
    stdOASprob.run()

    print('Run coupled system analysis with fixed point iteration')
    # Make local functions for coupled system analysis
    def f_aero(def_mesh, surface):
        loads = aerodynamics(def_mesh, surface, OASprob.prob_dict, OASprob.comp_dict)
        # loads = aerodynamics2(def_mesh, OASprob.surfaces[0], OASprob.OASprob.prob_dict)
        return loads
    def f_struct(loads, surface):
        def_mesh = structures(loads, surface, OASprob.prob_dict, OASprob.comp_dict)
        # def_mesh = structures2(loads, OASprob.surfaces[0], OASprob.OASprob.prob_dict)
        return def_mesh

    # Define FPI parameters
    utol = fpi_opt['utol']
    maxiter = fpi_opt['maxiter']
    # Use initial mesh with zero deformation
    surface = OASprob.surfaces[0]
    x0 = def_mesh = surface['def_mesh']
    print('def_mesh.dtype=',def_mesh.dtype)
    # x0 = f_aero(def_mesh, surface)*0.0
    u_norm = 1.0e99
    iter_count = 0
    # Run fixed point iteration on coupled aerodynamics-structures system
    while (iter_count < maxiter) and (u_norm > utol):
          # Update iteration counter
          iter_count += 1
          # Run iteration and evaluate norm of residual
          loads = aerodynamics(x0, surface, OASprob.prob_dict, OASprob.comp_dict)
          def_mesh = x = structures(loads, surface, OASprob.prob_dict, OASprob.comp_dict)
          u_norm = np.linalg.norm(x - x0)
          x0 = x

    if iter_count >= maxiter or math.isnan(u_norm):
        msg = 'FAILED to converge after {0:d} iterations'.format(iter_count)
    else:
        msg = 'Converged in {0:d} iterations'.format(iter_count)
    print(msg)

    PRINT('def_mesh=',def_mesh.real)
    PRINT('stdOASprob def_mesh=',stdOASprob.prob['coupled.wing.def_mesh'])
    PRINT('def_mesh error=',def_mesh.real-stdOASprob.prob['coupled.wing.def_mesh'])

    PRINT('loads=',loads.real)
    PRINT('stdOASprob loads=',stdOASprob.prob['coupled.wing.loads'])
    PRINT('loads error=',loads.real-stdOASprob.prob['coupled.wing.loads'])

    print('np.linalg.norm(def_mesh error)=',np.linalg.norm(def_mesh.real-stdOASprob.prob['coupled.wing.def_mesh']))
    print('np.linalg.norm(loads error)=',np.linalg.norm(loads.real-stdOASprob.prob['coupled.wing.loads']))

    print('Evaluate functional components...')
    aero_perf(surface, OASprob.prob_dict, OASprob.comp_dict)
    struct_perf(surface, OASprob.prob_dict, OASprob.comp_dict)
    # print(surface['CL'], surface['CD'], surface['weight'])
    fuelburn = functional_breguet_range([surface], surface['CL'], surface['CD'], surface['weight'], OASprob.prob_dict, OASprob.comp_dict['FunctionalBreguetRange'])
    eq_con = functional_equilibrium([surface], surface['L'], surface['weight'], fuelburn, OASprob.prob_dict, OASprob.comp_dict['FunctionalEquilibrium'])
    print('fuelburn=',fuelburn.real)
    print('stdOASprob.prob[fuelburn]=',stdOASprob.prob['fuelburn'])
    print('fuelburn abserr=',round(fuelburn.real-stdOASprob.prob['fuelburn'],3),
          'relerr={}%'.format(round(abs(fuelburn.real-stdOASprob.prob['fuelburn'])/stdOASprob.prob['fuelburn']*100,3),'%'))
