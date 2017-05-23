from __future__ import print_function
import analysis
from run_classes import OASProblem
import numpy as np
import sys



def test_aero(surface):
    print('Test aerodynamics')
    prob_dict = {}
    prob_dict['type'] = 'aero'

    # Using run_classes.py objects
    print('--- Use run_classes.py ---')
    rcOAS = OASProblem(prob_dict)
    rcOAS.add_surface(surface)
    rcMesh = rcOAS.surfaces[0]['mesh']
    rcOAS.setup()
    # print('rcOAS.prob.root._subsystems:',rcOAS.prob.root._subsystems)
    rcOAS.run()

    # Using analysis.py objects and functions
    print('--- Use analysis.py ---')
    anOAS = analysis.setup(prob_dict, [surface])
    anMesh = anOAS.surfaces[0]['mesh']
    # Generate initial mesh
    mesh = analysis.geometry_mesh(anOAS.surfaces[0], anOAS.comp_dict['GeometryMesh'])
    disp = np.zeros((anOAS.surfaces[0]['num_y'], 6), dtype=analysis.data_type)  # zero displacement
    def_mesh = analysis.transfer_displacements(mesh, disp, anOAS.comp_dict['TransferDisplacements'])
    print('...after setup functions but before GeometryMesh()...')
    print('anMESH=',anMesh)
    print('rcMESH=',rcMesh)
    print('errorMESH=',anMesh-rcMesh)
    print('...after GeometryMesh()...')
    print('anMESH=',mesh)
    print('rcMESH=',rcOAS.prob['wing.mesh'])
    print('errorMESH=',mesh-rcOAS.prob['wing.mesh'])
    # print('**def_mesh=',def_mesh)
    # print('**rcOAS.prob[wing.def_mesh]=',rcOAS.prob['wing.def_mesh'])
    # anOAS.surfaces[0].update({
    #     'mesh': mesh,
    #     'disp': disp,
    #     'def_mesh': def_mesh
    # })
    # anOAS.surfaces[0]['def_mesh'] = analysis.gen_init_mesh(anOAS.surfaces[0], anOAS.comp_dict)
    # v = anOAS.prob_dict.get('v')
    # alpha = anOAS.prob_dict.get('alpha')
    # size = anOAS.prob_dict.get('tot_panels')
    # rho = anOAS.prob_dict.get('rho')
    # def_mesh = anOAS.surfaces[0]['def_mesh']
    # b_pts, c_pts, widths, cos_sweep, lengths, normals, S_ref = analysis.vlm_geometry(def_mesh, anOAS.comp_dict['VLMGeometry'])
    # AIC, rhs= analysis.assemble_aic(anOAS.surfaces[0], def_mesh, b_pts, c_pts, normals, v, alpha,  anOAS.comp_dict['AssembleAIC'])
    # circulations = analysis.aero_circulations(AIC, rhs,  anOAS.comp_dict['AeroCirculations'])
    # sec_forces = analysis.vlm_forces(anOAS.surfaces[0], def_mesh, b_pts, circulations, alpha, v, rho,  anOAS.comp_dict['VLMForces'])
    # loads = analysis.transfer_loads(def_mesh, sec_forces,  anOAS.comp_dict['TransferLoads'])
    # # store variables in surface dict
    # varDict = {
    #     'b_pts': b_pts,
    #     'c_pts': c_pts,
    #     'widths': widths,
    #     'cos_sweep': cos_sweep,
    #     'lengths': lengths,
    #     'normals': normals,
    #     'S_ref': S_ref,
    #     'loads': loads,
    #     'sec_forces': sec_forces
    # }
    # anOAS.surfaces[0].update(varDict)
    # anOAS.prob_dict.update({
    #     'sec_forces': sec_forces
    # })
    # ANLY = anOAS.surfaces[0]
    # ANLY.update(anOAS.prob_dict)
    # RNCL = rcOAS.prob
    # prefix = surface['name']+'.'
    # print('  -----  ACCURACY  -----   ')
    # print('variable   |             error   ')
    # print('----------------------------------------------')
    # varList = ['mesh','def_mesh','b_pts','c_pts','widths','cos_sweep','lengths','normals','S_ref']
    # for var in varList:
    #     print('{:11s}|{}'.format(var,ANLY[var].real-RNCL[prefix+var]))
    # print(anOAS.surfaces[0]['fem_origin'])
    # print(rcOAS.surfaces[0]['fem_origin'])

    # var = 'disp'
    # print('{:11s}|{}'.format(var,ANLY[var].real-RNCL[var]))
#
#     return loads
#
#
#
# OASprob = analysis.setup(prob_dict, [surface])
#
# # Using standard method in run_classes.py
# stdOASprob = OASProblem(prob_dict)
# stdOASprob.add_surface(surface)
# stdOASprob.setup()
# stdOASprob.run()
#
# print('Run coupled system analysis with fixed point iteration')
# # Make local functions for coupled system analysis
# def f_aero(def_mesh, surface):
#     loads = analysis.aerodynamics(def_mesh, surface, OASprob.prob_dict, OASprob.comp_dict)
#     # loads = aerodynamics2(def_mesh, OASprob.surfaces[0], OASprob.OASprob.prob_dict)
#     return loads
# def f_struct(loads, surface):
#     def_mesh = analysis.structures(loads, surface, OASprob.prob_dict, OASprob.comp_dict)
#     # def_mesh = structures2(loads, OASprob.surfaces[0], OASprob.OASprob.prob_dict)
#     return def_mesh
#
# # Define FPI parameters
# utol = fpi_opt['utol']
# maxiter = fpi_opt['maxiter']
# # Generate initial mesh with zero deformation
# def_mesh = gen_init_mesh(OASprob.surfaces[0], OASprob.comp_dict)
# OASprob.surfaces[0]['def_mesh'] = def_mesh
# surface = OASprob.surfaces[0]
# x0 = f_aero(def_mesh, surface)*0.0
# # x0 = np.zeros((f_aero(def_mesh,surface).size))
# print(x0)
# u_norm = 1.0e99
# iter_count = 0
# # Run fixed point iteration on coupled aerodynamics-structures system
# while (iter_count < maxiter) and (u_norm > utol):
#       # Update iteration counter
#       iter_count += 1
#       # Run iteration and evaluate norm of residual
#       loads = x = aerodynamics(x0, surface, OASprob.prob_dict, OASprob.comp_dict)
#       def_mesh = structures(loads, surface, OASprob.prob_dict, OASprob.comp_dict)
#       u_norm = np.linalg.norm(x - x0)
#       x0 = x
#
# if iter_count >= maxiter or math.isnan(u_norm):
#     msg = 'FAILED to converge after {0:d} iterations'.format(iter_count)
# else:
#     msg = 'Converged in {0:d} iterations'.format(iter_count)
#
# print(msg)
#
# print('  -----  TEST ACCURACY  -----   ')
# print('variable  |   analysis.py   |   run_classes.py')
# print('----------------------------------------------')
# var = 'b_pts'
# print('{0:9s}|{a[b_pts]}|{b[coupled.wing.b_pts]}'.format(var,a=surface,b=stdOASprob.prob))
# # b_pts, c_pts, widths, cos_sweep, lengths, normals, S_ref = vlm_geometry(def_mesh, comp_dict['VLMGeometry'])
# # AIC, rhs= assemble_aic(surface, def_mesh, b_pts, c_pts, normals, v, alpha, comp_dict['AssembleAIC'])
# # circulations = aero_circulations(AIC, rhs, comp_dict['AeroCirculations'])
# # sec_forces = vlm_forces(surface, def_mesh, b_pts, circulations, alpha, v, rho, comp_dict['VLMForces'])
# # loads = transfer_loads(def_mesh, sec_forces, comp_dict['TransferLoads'])
#
#
# print('def_mesh=\n',def_mesh.real)
# print('stdOASprob def_mesh=\n',stdOASprob.prob['coupled.wing.def_mesh'])
# print('def_mesh error=\n',def_mesh.real-stdOASprob.prob['coupled.wing.def_mesh'])
# print('np.linalg.norm(def_mesh error)=',np.linalg.norm(def_mesh.real-stdOASprob.prob['coupled.wing.def_mesh']))
#
#
# print('loads=\n',loads.real)
# print('stdOASprob loads=\n',stdOASprob.prob['coupled.wing.loads'])
# print('loads error=\n',loads.real-stdOASprob.prob['coupled.wing.loads'])
# print('np.linalg.norm(loads error)=',np.linalg.norm(loads.real-stdOASprob.prob['coupled.wing.loads']))
#
# print('Evaluate functional components...')
# analysis.aero_perf(surface, OASprob.prob_dict, OASprob.comp_dict)
# analysis.struct_perf(surface, OASprob.prob_dict, OASprob.comp_dict)
# # print(surface['CL'], surface['CD'], surface['weight'])
# fuelburn = analysis.functional_breguet_range([surface], surface['CL'], surface['CD'], surface['weight'], OASprob.prob_dict, OASprob.comp_dict['FunctionalBreguetRange'])
# eq_con = analysis.functional_equilibrium([surface], surface['L'], surface['weight'], fuelburn, OASprob.prob_dict, OASprob.comp_dict['FunctionalEquilibrium'])
# print('fuelburn=',fuelburn.real)
# print('stdOASprob.prob[fuelburn]=',stdOASprob.prob['fuelburn'])

if __name__ == "__main__":
    # Define surface
    surface = {
        'name': 'wing',
        'wing_type' : 'CRM',
        'num_x': 2,   # number of chordwise points
        'num_y': 9    # number of spanwise points
    }
    if (not len(sys.argv)>1) or (sys.argv[1] == 'aero'):
        test_aero(surface)
