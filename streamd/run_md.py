import argparse
import logging
import os
import shutil
import subprocess
from datetime import datetime
from functools import partial
from glob import glob
from multiprocessing import cpu_count

from streamd.md_analysis import run_md_analysis
from streamd.preparation.complex_preparation import run_complex_preparation
from streamd.preparation.ligand_preparation import prepare_input_ligands, check_mols
from streamd.utils.dask_init import init_dask_cluster, calc_dask
from streamd.utils.utils import filepath_type, run_check_subprocess


class RawTextArgumentDefaultsHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def run_equilibration(wdir, project_dir, bash_log):
    if os.path.isfile(os.path.join(wdir, 'npt.gro')) and os.path.isfile(os.path.join(wdir, 'npt.cpt')):
        logging.warning(f'{wdir}. Checkpoint files after Equilibration exist. '
                        f'Equilibration step will be skipped ')
        return wdir

    try:
        subprocess.check_output(f'wdir={wdir} bash {os.path.join(project_dir, "scripts/script_sh/equlibration.sh")}'
                                f'>> {bash_log} 2>&1',
                                shell=True)
    except subprocess.CalledProcessError as e:
        logging.exception(f'{wdir}\n{e}', stack_info=True)
        return None
    return wdir


def run_simulation(wdir, project_dir, bash_log):
    if os.path.isfile(os.path.join(wdir, 'md_out.tpr')) and os.path.isfile(os.path.join(wdir, 'md_out.cpt')) \
            and os.path.isfile(os.path.join(wdir, 'md_out.xtc')) and os.path.isfile(os.path.join(wdir, 'md_out.log')):
        logging.warning(f'{wdir}. md_out.xtc and md_out.tpr exist. '
                        f'MD simulation step will be skipped. '
                        f'You can rerun the script and use --wdir_to_continue {wdir} --md_time time_in_ns to extend current trajectory.')
        return wdir
    try:
        subprocess.check_output(f'wdir={wdir} bash {os.path.join(project_dir, "scripts/script_sh/md.sh")}'
                                f'>> {bash_log} 2>&1', shell=True)
    except subprocess.CalledProcessError as e:
        logging.exception(f'{wdir}\n{e}', stack_info=True)
        return None
    return wdir


def continue_md_from_dir(wdir_to_continue, tpr, cpt, xtc, deffnm_prev, deffnm_next, mdtime_ns, project_dir, bash_log):
    def continue_md(tpr, cpt, xtc, wdir, new_mdtime_ps, deffnm_next, project_dir, bash_log):
        try:
            subprocess.check_output(f'wdir={wdir} tpr={tpr} cpt={cpt} xtc={xtc} new_mdtime_ps={new_mdtime_ps} '
                                    f'deffnm_next={deffnm_next} bash {os.path.join(project_dir, "scripts/script_sh/continue_md.sh")}'
                                    f'>> {bash_log} 2>&1',
                                    shell=True)
        except subprocess.CalledProcessError as e:
            logging.exception(f'{wdir}\n{e}', stack_info=True)
            return None

        return wdir

    if tpr is None:
        tpr = os.path.join(wdir_to_continue, f'{deffnm_prev}.tpr')
    if cpt is None:
        cpt = os.path.join(wdir_to_continue, f'{deffnm_prev}.cpt')
    if xtc is None:
        xtc = os.path.join(wdir_to_continue, f'{deffnm_prev}.xtc')

    for i in [tpr, cpt, xtc]:
        if not os.path.isfile(i):
            logging.exception(
                f'No {i} file was found. Cannot continue the simulation. Calculations will be interrupted ',
                stack_info=True)
            return None

    new_mdtime_ps = int(mdtime_ns * 1000)

    # check previous existing files with the same name
    for f in glob(os.path.join(wdir_to_continue, f'{deffnm_next}*')):
        n = len(glob(os.path.join(wdir_to_continue, f'#{os.path.basename(f)}.*#'))) + 1
        new_f = os.path.join(wdir_to_continue, f'#{os.path.basename(f)}.{n}#')
        shutil.move(f, new_f)
        logging.warning(f'Backup previous file {f} to {new_f}')

    return continue_md(tpr=tpr, cpt=cpt, xtc=xtc, wdir=wdir_to_continue,
                       new_mdtime_ps=new_mdtime_ps, deffnm_next=deffnm_next, project_dir=project_dir, bash_log=bash_log)


def start(protein, wdir, lfile, system_lfile,
          clean_previous, mdtime_ns, npt_time_ps, nvt_time_ps,
          topol, topol_itp_list, posre_list_protein,
          wdir_to_continue_list, deffnm_prev,
          tpr_prev, cpt_prev, xtc_prev,
          activate_gaussian, gaussian_exe,
          hostfile, ncpu, not_clean_log_files,
          forcefield_num=6, bash_log=None):
    '''
    :param protein: protein file - pdb or gro format
    :param wdir: None or path
    :param lfile: None or file
    :param system_lfile: None or file. Mol or sdf format
    :param forcefield_num: int
    :param clean_previous: boolean. Remove all previous md files
    :param mdtime_ns: float. Time in ns
    :param npt_time_ps: int. Time in ps
    :param nvt_time_ps: int. Time in ps
    :param topol: None or file
    :param topol_itp_list: None or list of files
    :param posre_list_protein: None or list of files
    :param wdir_to_continue_list: list of paths
    :param tpr_prev: None or file
    :param cpt_prev: None or file
    :param xtc_prev: None or file
    :param deffnm_prev: md_out
    :param hostfile: None or file
    :param ncpu:
    not_clean_log_files: boolean. Remove backup md files (starts with #)
    :return:
    '''

    project_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_dir, 'scripts')
    script_mdp_path = os.path.join(script_path, 'mdp')

    if wdir_to_continue_list is None:
        # create dirs
        wdir_protein = os.path.join(wdir, 'md_files', 'md_preparation', 'protein')
        wdir_ligand = os.path.join(wdir, 'md_files', 'md_preparation', 'var_lig')
        wdir_system_ligand = os.path.join(wdir, 'md_files', 'md_preparation', 'system_lig')

        wdir_md = os.path.join(wdir, 'md_files', 'md_run')

        os.makedirs(wdir_md, exist_ok=True)
        os.makedirs(wdir_protein, exist_ok=True)
        os.makedirs(wdir_ligand, exist_ok=True)
        os.makedirs(wdir_system_ligand, exist_ok=True)

        if not os.path.isfile(protein):
            raise FileExistsError(f'{protein} does not exist')

        pname, p_ext = os.path.splitext(os.path.basename(protein))
        # check if already exist in the working directory
        if not os.path.isfile(f'{os.path.join(wdir_protein, pname)}.gro') or not os.path.isfile(
                os.path.join(wdir_protein, "topol.top")):
            if p_ext != '.gro' or topol is None or posre_list_protein is None:
                try:
                    logging.info('Start protein preparation')
                    subprocess.check_output(
                        f'gmx pdb2gmx -f {protein} -o {os.path.join(wdir_protein, pname)}.gro -water tip3p -ignh '
                        f'-i {os.path.join(wdir_protein, "posre.itp")} '
                        f'-p {os.path.join(wdir_protein, "topol.top")}'
                        f'<<< {forcefield_num}'
                        f'>> {bash_log} 2>&1', shell=True)
                    logging.info(f'Successfully finished protein preparation\n')
                except subprocess.CalledProcessError as e:
                    logging.exception(e, stack_info=True)
                    return None
            else:
                target_path = os.path.join(wdir_protein, os.path.basename(protein))
                if not os.path.isfile(target_path):
                    shutil.copy(protein, target_path)
                target_path = os.path.join(wdir_protein, 'topol.top')
                if not os.path.isfile(target_path):
                    shutil.copy(topol, target_path)
                # multiple chains
                for posre_protein in posre_list_protein:
                    target_path = os.path.join(wdir_protein, os.path.basename(posre_protein))
                    if not os.path.isfile(target_path):
                        shutil.copy(posre_protein, target_path)
                if topol_itp_list is not None:
                    if len(posre_list_protein) != len(topol_itp_list):
                        logging.exception('The number of protein_chainX.itp files should be equal the number of posre_protein_chainX.itp files.\n'
                                          'Check --topol_itp and --posre arguments')
                        return None
                    for topol_itp in topol_itp_list:
                        target_path = os.path.join(wdir_protein, os.path.basename(topol_itp))
                        if not os.path.isfile(target_path):
                            shutil.copy(topol_itp, target_path)


        else:
            logging.warning(f'{os.path.join(wdir_protein, pname)}.gro and topol.top files exist. '
                            f'Protein preparation step will be skipped.')

        # Part 1. Ligand Preparation
        if system_lfile is not None:
            logging.info('Start cofactor preparation')
            number_of_mols, problem_mols = check_mols(system_lfile)
            if problem_mols:
                logging.exception(f'Cofactor molecules: {problem_mols} from {system_lfile} cannot be processed. Script will be interrupted.')
                return None

            system_lig_wdirs = prepare_input_ligands(system_lfile, preset_resid=None, script_path=script_path,
                                                     project_dir=project_dir, wdir_ligand=wdir_system_ligand,
                                                     gaussian_exe=gaussian_exe, activate_gaussian=activate_gaussian,
                                                     hostfile=hostfile, ncpu=ncpu, bash_log=bash_log)
            if number_of_mols != len(system_lig_wdirs):
                logging.exception(f'Error with cofactor preparation. Only {len(system_lig_wdirs)} from {number_of_mols} preparation were finished.'
                                  f' The calculation will be interrupted')
                return None

            logging.info(f'Successfully finished {len(system_lig_wdirs)} cofactor preparation\n')
        else:
            system_lig_wdirs = []  # wdir_ligand_cur

        if lfile is not None:
            logging.info('Start ligand preparation')
            number_of_mols, problem_mols = check_mols(lfile)
            if problem_mols:
                logging.warning(f'Ligand molecules: {problem_mols} from {lfile} cannot be processed.'
                                f' Such molecules will be skipped.')

            var_lig_wdirs = prepare_input_ligands(lfile, preset_resid='UNL', script_path=script_path,
                                                  project_dir=project_dir, wdir_ligand=wdir_ligand,
                                                  gaussian_exe=gaussian_exe, activate_gaussian=activate_gaussian,
                                                  hostfile=hostfile, ncpu=ncpu, bash_log=bash_log)
            if number_of_mols != len(var_lig_wdirs):
                logging.warning(f'Problem with ligand preparation. Only {len(var_lig_wdirs)} from {number_of_mols} preparation were finished.'
                                f' Such molecules will be skipped.')

            logging.info(f'Successfully finished {len(var_lig_wdirs)} ligand preparation\n')
        else:
            var_lig_wdirs = [[]]  # run protein in water only simulation

            # make all.itp and create complex
            logging.info('Start complex preparation')
            var_complex_prepared_dirs = []

            for res in calc_dask(run_complex_preparation, var_lig_wdirs, dask_client,
                                 wdir_system_ligand_list=system_lig_wdirs,
                                 protein_name=pname, wdir_protein=wdir_protein,
                                 clean_previous=clean_previous, wdir_md=wdir_md,
                                 script_path=script_mdp_path, project_dir=project_dir, mdtime_ns=mdtime_ns,
                                 npt_time_ps=npt_time_ps, nvt_time_ps=nvt_time_ps, bash_log=bash_log):
                if res:
                    var_complex_prepared_dirs.append(res)
            logging.info(f'Successfully finished {len(var_complex_prepared_dirs)} complex preparation\n')

        finally:
            dask_client.shutdown()
            if cluster:
                cluster.close()

        # Part 2. Equilibration and MD simulation. Run on all cpu
        dask_client, cluster = init_dask_cluster(hostfile=hostfile, n_tasks_per_node=1, ncpu=ncpu)
        try:
            logging.info('Start Equilibration steps')
            var_eq_dirs = []
            for res in calc_dask(run_equilibration, var_complex_prepared_dirs, dask_client, project_dir=project_dir, bash_log=bash_log):
                if res:
                    var_eq_dirs.append(res)
            logging.info(f'Successfully finished {len(var_eq_dirs)} Equilibration step\n')

            var_md_dirs = []
            logging.info('Start Simulation step')
            for res in calc_dask(run_simulation, var_eq_dirs, dask_client, project_dir=project_dir, bash_log=bash_log):
                if res:
                    var_md_dirs.append(res)

        finally:
            logging.warning('Oook. finishing. shutdown')
            dask_client.shutdown()
            if cluster:
                cluster.close()

        deffnm = 'md_out'
        logging.info(f'Simulation of {len(var_md_dirs)} were successfully finished\nFinished: {var_md_dirs}\n')

    else:  # continue prev md
        dask_client, cluster = init_dask_cluster(hostfile=hostfile, n_tasks_per_node=1, ncpu=ncpu)
        try:
            logging.info('Start Continue Simulation step')
            var_md_dirs = []
            deffnm = f'{deffnm_prev}_{mdtime_ns}'
            # os.path.dirname(var_lig)
            for res in calc_dask(continue_md_from_dir, wdir_to_continue_list, dask_client,
                                 tpr=tpr_prev, cpt=cpt_prev, xtc=xtc_prev,
                                 deffnm_prev=deffnm_prev, deffnm_next=deffnm, mdtime_ns=mdtime_ns,
                                 project_dir=project_dir, bash_log=bash_log):
                if res:
                    var_md_dirs.append(res)

        finally:
            dask_client.shutdown()
            if cluster:
                cluster.close()
        logging.info(f'Continue of simulation of {len(var_md_dirs)} were successfully finished\nFinished: {var_md_dirs}\n')

    # Part 3. MD Analysis. Run on each cpu
    dask_client, cluster = init_dask_cluster(hostfile=hostfile, n_tasks_per_node=ncpu, ncpu=ncpu)
    try:
        logging.info('Start Analysis of the simulations')
        var_md_analysis_dirs = []
        # os.path.dirname(var_lig)
        for res in calc_dask(run_md_analysis, var_md_dirs,
                             dask_client, deffnm=deffnm, mdtime_ns=mdtime_ns, project_dir=project_dir, bash_log=bash_log):
            if res:
                var_md_analysis_dirs.append(res)
    finally:
        dask_client.shutdown()
        if cluster:
            cluster.close()

    logging.info(f'Analysis of md simulation of {len(var_md_analysis_dirs)} were successfully finished\nFinished: {var_md_analysis_dirs}')

    if not not_clean_log_files:
        if wdir_to_continue_list is None:
            for f in glob(os.path.join(wdir_md, '*', '#*#')):
                os.remove(f)
        else:
            for wdir_md in wdir_to_continue_list:
                for f in glob(os.path.join(wdir_md, '#*#')):
                    os.remove(f)

def main():
    parser = argparse.ArgumentParser(description='''Run or continue MD simulation.\n
    Allowed systems: Protein, Protein-Ligand, Protein-Cofactors(multiple), Protein-Ligand-Cofactors(multiple) ''')
    parser.add_argument('-p', '--protein', metavar='FILENAME', required=False,
                        type=partial(filepath_type, ext=('pdb', 'gro'), check_exist=True),
                        help='input file of protein. Supported formats: *.pdb or gro')
    parser.add_argument('-d', '--wdir', metavar='WDIR', default=None,
                        type=partial(filepath_type, check_exist=False, create_dir=True),
                        help='Working directory. If not set the current directory will be used.')
    parser.add_argument('-l', '--ligand', metavar='FILENAME', required=False,
                        type=partial(filepath_type, ext=('mol', 'sdf')),
                        help='input file with compound. Supported formats: *.mol or sdf')
    parser.add_argument('--cofactor', metavar='FILENAME', default=None,
                        type=partial(filepath_type, ext=('mol', 'sdf')),
                        help='input file with compound. Supported formats: *.mol or sdf')
    parser.add_argument('--clean_previous_md', action='store_true', default=False,
                        help='remove a production MD simulation directory if it exists to re-initialize production MD setup')
    parser.add_argument('--hostfile', metavar='FILENAME', required=False, type=str, default=None,
                        help='text file with addresses of nodes of dask SSH cluster. The most typical, it can be '
                             'passed as $PBS_NODEFILE variable from inside a PBS script. The first line in this file '
                             'will be the address of the scheduler running on the standard port 8786. If omitted, '
                             'calculations will run on a single machine as usual.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, default=cpu_count(), type=int,
                        help='number of CPU per server. Use all cpus by default.')
    parser.add_argument('--topol', metavar='topol.top', required=False, default=None, type=filepath_type,
                        help='topology file (required if a gro-file is provided for the protein).'
                             'All output files obtained from gmx2pdb should preserve the original names')
    parser.add_argument('--topol_itp', metavar='topol_chainA.itp topol_chainB.itp', required=False, nargs='+',
                        default=None, type=filepath_type,
                        help='Itp files for individual protein chains (required if a gro-file is provided for the protein).'
                             'All output files obtained from gmx2pdb should preserve the original names')
    parser.add_argument('--posre', metavar='posre.itp', required=False, nargs='+', default=None, type=filepath_type,
                        help='posre file(s) (required if a gro-file is provided for the protein).'
                             'All output files obtained from gmx2pdb should preserve the original names')
    parser.add_argument('--md_time', metavar='ns', required=False, default=1, type=float,
                        help='time of MD simulation in ns')
    parser.add_argument('--npt_time', metavar='ps', required=False, default=100, type=int,
                        help='time of NPT equilibration in ps')
    parser.add_argument('--nvt_time', metavar='ps', required=False, default=100, type=int,
                        help='time of NVT equilibration in ps')
    parser.add_argument('--not_clean_log_files', action='store_true', default=False,
                        help='Not to remove all backups of md files')
    # continue md
    parser.add_argument('--tpr', metavar='FILENAME', required=False, default=None, type=filepath_type,
                        help='tpr file from the previous MD simulation')
    parser.add_argument('--cpt', metavar='FILENAME', required=False, default=None, type=filepath_type,
                        help='cpt file from previous simulation')
    parser.add_argument('--xtc', metavar='FILENAME', required=False, default=None, type=filepath_type,
                        help='xtc file from previous simulation')
    parser.add_argument('--wdir_to_continue', metavar='DIRNAME', required=False, default=None, nargs='+',
                        type=partial(filepath_type, exist_type='dir'),
                        help='''directories for the previous simulations. Use to extend or continue the simulation. '
                             Should consist of: tpr, cpt, xtc files''')
    parser.add_argument('--deffnm', metavar='preffix for md files', required=False, default='md_out',
                        help='''preffix for the previous md files. Use to extend or continue the simulation.
                        Only if wdir_to_continue is used. Use if each --tpr, --cpt, --xtc arguments are not set up. 
                        Files deffnm.tpr, deffnm.cpt, deffnm.xtc will be used from wdir_to_continue''')
    parser.add_argument('--activate_gaussian', metavar='module load Gaussian/09-d01', required=False, default=None,
                        help='string that load gaussian module if ')
    parser.add_argument('--gaussian_exe', metavar='g09 or /apps/all/Gaussian/09-d01/g09/g09', required=False,
                        default=None,
                        help='path to gaussian executable or alias')

    args = parser.parse_args()

    if args.wdir is None:
        wdir = os.getcwd()
    else:
        wdir = args.wdir

    out_time = f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'
    log_file = os.path.join(wdir,
                            f'log_{os.path.basename(str(args.protein))[:-4]}_{os.path.basename(str(args.ligand))[:-4]}_{os.path.basename(str(args.cofactor))[:-4]}_'
                            f'{out_time}.log')
    bash_log = os.path.join(wdir,f'streamd_bash_{out_time}.log')

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    logging.getLogger('distributed').setLevel('WARNING')
    logging.getLogger('asyncssh').setLevel('WARNING')
    logging.getLogger('distributed.worker').setLevel('WARNING')
    logging.getLogger('distributed.core').setLevel('WARNING')
    logging.getLogger('distributed.comm').setLevel('WARNING')
    logging.getLogger('distributed.nanny').setLevel('CRITICAL')
    logging.getLogger('bockeh').setLevel('WARNING')

    logging.info(args)
    try:
        start(protein=args.protein, lfile=args.ligand,
              clean_previous=args.clean_previous_md, system_lfile=args.cofactor,
              topol=args.topol, topol_itp_list=args.topol_itp, posre_list_protein=args.posre, mdtime_ns=args.md_time,
              npt_time_ps=args.npt_time, nvt_time_ps=args.nvt_time,
              wdir_to_continue_list=args.wdir_to_continue, deffnm_prev=args.deffnm,
              tpr_prev=args.tpr, cpt_prev=args.cpt, xtc_prev=args.xtc,
              activate_gaussian=args.activate_gaussian, gaussian_exe=args.gaussian_exe,
              hostfile=args.hostfile, ncpu=args.ncpu, wdir=wdir, not_clean_log_files=args.not_clean_log_files,
              bash_log=bash_log)
    finally:
        logging.shutdown()