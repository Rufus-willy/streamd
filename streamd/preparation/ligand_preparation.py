"""Routines for preparing ligand parameters and inputs."""

import itertools
import logging
import os

from glob import glob
import re

import parmed as pmd

from rdkit import Chem
from rdkit.Chem import rdmolops

from streamd.utils.dask_init import init_dask_cluster, calc_dask
from streamd.utils.utils import run_check_subprocess

def reorder_hydrogens(mol):
    """Move hydrogens to follow their heavy atom in atom order."""
    new_order = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:  # Not a hydrogen
            new_order.append(atom.GetIdx())
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # Is a hydrogen
                    new_order.append(neighbor.GetIdx())
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return mol

def supply_mols_tuple(fname, preset_resid=None, protein_resid_set=None):
    """Yield RDKit molecules with generated residue identifiers."""
    def generate_resid(protein_resid_set):
        ascii_uppercase_digits = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in itertools.product(ascii_uppercase_digits, repeat=3):
            resid = ''.join(i)
            if resid != 'UNL' and (protein_resid_set is None or resid not in protein_resid_set):
                yield resid

    def add_ids(mol, n, input_fname, resid):
        mol.SetProp('resid', resid)
        if not mol.HasProp('_Name') or not mol.GetProp('_Name'):
            mol.SetProp('_Name', f'{input_fname}_{n}')
        return mol

    resid_generator = generate_resid(protein_resid_set=protein_resid_set)

    if fname.endswith('.sdf'):
        for n, mol in enumerate(Chem.SDMolSupplier(fname, removeHs=False), 1):
            if mol:
                if preset_resid is None:
                    resid = next(resid_generator)
                else:
                    resid = preset_resid

                mol = add_ids(mol, n, input_fname=os.path.basename(fname).rstrip('.sdf'), resid=resid)
                yield (mol, mol.GetProp('_Name'), resid)

    if fname.endswith('.mol') or fname.endswith('.mol2'):
        if fname.endswith('.mol2'):
            mol = pmd.load_file(fname).to_structure()
            mol = mol.rdkit_mol
        else:
            mol = Chem.MolFromMolFile(fname, removeHs=False)

        if mol:
            if preset_resid is None:
                resid = next(resid_generator)
            else:
                resid = preset_resid
            mol = add_ids(mol, n=1, input_fname=os.path.basename(fname).rstrip('.mol2').rstrip('.mol'), resid=resid)
            yield (mol, mol.GetProp('_Name'), resid)


def check_mols(fname):
    """Return number of molecules and list of problematic ones.
    :param fname:
    :return: number of mols, list of problem_mols
    """
    def check_if_problem(mol, n):
        molid = mol.GetProp('_Name') if mol.HasProp('_Name') else n
        try:
            Chem.SanitizeMol(mol)
        except:
            return molid
        return None
        
    problem_mols = []
    number_of_mols = 0
    if fname.endswith('.sdf'):
        n = 0
        for n, mol in enumerate(Chem.SDMolSupplier(fname, removeHs=False, sanitize=False), 1):
            problem_molid = check_if_problem(mol, n)
            if problem_molid:
                problem_mols.append(problem_molid)
        number_of_mols = n

    if fname.endswith('.mol') or fname.endswith('.mol2'):
        if fname.endswith('.mol2'):
            mol = pmd.load_file(fname).to_structure()
            mol = mol.rdkit_mol
        else:
            mol = Chem.MolFromMolFile(fname, removeHs=False, sanitize=False)
        problem_molid = check_if_problem(mol, 1)
        if problem_molid:
            problem_mols.append(problem_molid)
        number_of_mols = 1

    return number_of_mols, problem_mols


def make_all_itp(fileitp_input_list, fileitp_output_list, out_file):
    """Merge ligand ITP files and collect unique atom types."""
    atom_type_list = []
    start_columns = None
    # '[ atomtypes ]\n; name    at.num    mass    charge ptype  sigma      epsilon\n'
    for itp_input, itp_output in zip(fileitp_input_list, fileitp_output_list):
        with open(itp_input) as input:
            data = input.read()
        start = data.find('[ atomtypes ]')
        end = data.find('[ moleculetype ]') - 1
        atom_type_list.extend(data[start:end].split('\n')[2:])
        if start_columns is None:
            start_columns = data[start:end].split('\n')[:2]
        new_data = data[:start] + data[end + 1:]
        with open(itp_output, 'w') as output:
            output.write(new_data)

    atom_type_uniq = [i for i in set(atom_type_list) if i]
    with open(out_file, 'w') as output:
        output.write('\n'.join(start_columns) + '\n')
        output.write('\n'.join(atom_type_uniq) + '\n')

def prepare_tleap(tleap_template, tleap, molid, conda_env_path):
    """Fill a tleap template with ligand identifiers and environment path."""
    with open(tleap_template) as inp:
        data = inp.read()
    new_data = data.replace('env_path', conda_env_path).replace('ligand', molid)
    with open(tleap, 'w') as output:
        output.write(new_data)


def prepare_gaussian_files(file_template, file_out, ncpu, opt_restart=False, gaussian_basis=r'B3LYP/6-31G*',
                           gaussian_memory='60GB'):
    """Create Gaussian input from template adjusting resources and options."""
    with open(file_template) as inp:
        data = inp.read()
    standard_basis = r'B3LYP/6-31G\*'
    new_data = re.sub('%NProcShared=[0-9]*', f'%NProcShared={ncpu}', data)
    new_data = re.sub('%Mem=[0-9a-zA-Z]*', f'%Mem={gaussian_memory}', new_data)
    if 'SCF=' not in new_data:
        new_data = re.sub(standard_basis, f'{gaussian_basis} SCF=XQC', new_data)
    else:
        new_data = re.sub(standard_basis, gaussian_basis, new_data)
    if opt_restart and 'Opt' in new_data and 'Opt=Restart' not in new_data:
        new_data = re.sub('Opt', 'Opt=Restart', new_data)

    with open(file_out, 'w') as output:
        output.write(new_data)


def find_sobtop_dir(explicit_path=None):
    """Locate the Sobtop installation directory.

    Search order: explicit argument > SOBTOP_DIR env var > project sobtop/sobtop_*/ glob.
    Returns the directory path or None if not found.
    """
    def _is_sobtop_dir(d):
        return os.path.isfile(os.path.join(d, 'sobtop'))

    if explicit_path and os.path.isdir(explicit_path) and _is_sobtop_dir(explicit_path):
        return explicit_path

    env_dir = os.environ.get('SOBTOP_DIR')
    if env_dir and os.path.isdir(env_dir) and _is_sobtop_dir(env_dir):
        return env_dir

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in sorted(glob(os.path.join(project_root, 'sobtop', 'sobtop_*'))):
        if _is_sobtop_dir(d):
            return d

    return None


def _convert_mol2_types_to_elements(mol2_path, output_path, rdkit_mol):
    """Replace GAFF atom types with element symbols in a mol2 file.

    Sobtop requires element symbols (C, H, O) in the atom type column,
    but antechamber outputs GAFF types (ca, ha, o). This conversion uses
    the RDKit mol to supply correct element symbols while preserving charges.
    """
    elements = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    with open(mol2_path) as f:
        lines = f.readlines()

    in_atoms = False
    atom_idx = 0
    result = []
    for line in lines:
        s = line.strip()
        if s.startswith('@<TRIPOS>ATOM'):
            in_atoms = True
            result.append(line)
            continue
        elif s.startswith('@<TRIPOS>'):
            in_atoms = False
            result.append(line)
            continue
        if in_atoms and s and atom_idx < len(elements):
            parts = line.split()
            if len(parts) >= 6:
                parts[5] = elements[atom_idx]
                result.append(' '.join(parts) + '\n')
                atom_idx += 1
                continue
        result.append(line)

    with open(output_path, 'w') as f:
        f.writelines(result)


def inject_charges_to_itp(itp_path, mol2_path):
    """Read charges from a mol2 file and inject them into an itp file.

    Parses the @<TRIPOS>ATOM section of the mol2 (column 9 = charge) and
    replaces the charge column (column 7) in the [ atoms ] section of the itp.
    Raises ValueError if atom counts do not match.
    """
    # Read charges from mol2
    charges = []
    with open(mol2_path) as f:
        in_atoms = False
        for line in f:
            s = line.strip()
            if s.startswith('@<TRIPOS>ATOM'):
                in_atoms = True
                continue
            elif s.startswith('@<TRIPOS>'):
                in_atoms = False
                continue
            if in_atoms and s:
                parts = line.split()
                if len(parts) >= 9:
                    charges.append(float(parts[8]))

    # Read and modify itp
    with open(itp_path) as f:
        lines = f.readlines()

    in_atom_section = False
    atom_idx = 0
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped == '[ atoms ]':
            in_atom_section = True
            result.append(line)
            continue
        elif stripped.startswith('[') and stripped.endswith(']'):
            in_atom_section = False
            result.append(line)
            continue

        if in_atom_section and stripped and not stripped.startswith(';'):
            parts = line.split()
            if len(parts) >= 7:
                if atom_idx < len(charges):
                    parts[6] = f'{charges[atom_idx]:.8f}'
                result.append(' '.join(parts) + '\n')
                atom_idx += 1
                continue

        result.append(line)

    if atom_idx != len(charges):
        raise ValueError(
            f'Atom count mismatch: mol2 has {len(charges)} atoms, '
            f'itp [ atoms ] has {atom_idx} atoms')

    with open(itp_path, 'w') as f:
        f.writelines(result)


def prep_ligand_sobtop(mol_tuple, script_path, wdir_ligand, bash_log,
                       sobtop_dir, charge_method='gasteiger', ncpu=1,
                       no_dr=False, mol2_file=None, env=None):
    """Prepare ligand topology using Sobtop as the parameterization backend.

    Generates {molid}.itp, posre_{molid}.itp, {molid}.gro, and resid.txt
    with the same output interface as prep_ligand().
    """
    mol, molid, resid = mol_tuple
    wdir_ligand_cur = os.path.abspath(os.path.join(wdir_ligand, molid))
    os.makedirs(wdir_ligand_cur, exist_ok=True)

    itp_path = os.path.join(wdir_ligand_cur, f'{molid}.itp')
    posre_path = os.path.join(wdir_ligand_cur, f'posre_{molid}.itp')

    # Skip if already done
    if os.path.isfile(itp_path) and os.path.isfile(posre_path):
        logging.warning(
            f'{molid}.itp and posre_{molid}.itp already exist. '
            f'Sobtop preparation will be skipped for {molid}.')
        if not os.path.isfile(os.path.join(wdir_ligand_cur, 'resid.txt')):
            with open(os.path.join(wdir_ligand_cur, 'resid.txt'), 'w') as out:
                out.write(f'{molid}\t{resid}\n')
        return wdir_ligand_cur

    # Map charge method to antechamber -c flag
    ac_charge = 'gas' if charge_method == 'gasteiger' else 'bcc'

    mol2_path = os.path.join(wdir_ligand_cur, f'{molid}.mol2')

    if not mol2_file or not os.path.isfile(mol2_file):
        mol_file = os.path.join(wdir_ligand_cur, f'{molid}.mol')
        mol = Chem.AddHs(mol, addCoords=True)
        mol = reorder_hydrogens(mol)
        Chem.MolToMolFile(mol, mol_file)
        charge = rdmolops.GetFormalCharge(mol)

        # Generate mol2 with charges via antechamber
        cmd = (f'charge_method={ac_charge} lfile={mol_file} '
               f'input_dirname={wdir_ligand_cur} resid={resid} '
               f'molid={molid} charge={charge} dr=yes '
               f'bash {os.path.join(script_path, "script_sh", "ligand_mol2prep.sh")} '
               f'>> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1')

        success = run_check_subprocess(
            cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env,
            ignore_error=True if (no_dr or ac_charge == 'bcc') else False)

        if not success:
            if ac_charge == 'bcc':
                logging.warning(
                    f'AM1-BCC charge calculation failed for {molid}. '
                    f'Falling back to Gasteiger charges.')
                ac_charge = 'gas'
                cmd = (f'charge_method={ac_charge} lfile={mol_file} '
                       f'input_dirname={wdir_ligand_cur} resid={resid} '
                       f'molid={molid} charge={charge} dr=yes '
                       f'bash {os.path.join(script_path, "script_sh", "ligand_mol2prep.sh")} '
                       f'>> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1')
                success = run_check_subprocess(
                    cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env,
                    ignore_error=True if no_dr else False)

            if not success and not no_dr:
                return None
            elif not success:
                logging.warning(
                    f'antechamber failed for {molid} with dr=yes. Retrying with dr=no.')
                cmd = (f'charge_method={ac_charge} lfile={mol_file} '
                       f'input_dirname={wdir_ligand_cur} resid={resid} '
                       f'molid={molid} charge={charge} dr=no '
                       f'bash {os.path.join(script_path, "script_sh", "ligand_mol2prep.sh")} '
                       f'>> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1')
                if not run_check_subprocess(
                        cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env):
                    return None

        # Convert GAFF types to element symbols for Sobtop
        _convert_mol2_types_to_elements(mol2_path, mol2_path, mol)
    else:
        mol2 = pmd.load_file(mol2_file).to_structure()
        mol2.residues[0].name = 'UNL'
        mol2.save(mol2_path)
        logging.info(f'Using provided mol2 file: {mol2_file}')
        mol = mol2.rdkit_mol
        _convert_mol2_types_to_elements(mol2_path, mol2_path, mol)

    # Run Sobtop to generate topology and coordinates
    cmd = (f'input_dirname={wdir_ligand_cur} molid={molid} '
           f'sobtop_dir={sobtop_dir} nthreads=1 '
           f'bash {os.path.join(script_path, "script_sh", "ligand_sobtop.sh")} '
           f'>> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1')
    if not run_check_subprocess(
            cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env):
        return None

    # Inject charges from mol2 into itp (Sobtop writes zero charges)
    inject_charges_to_itp(itp_path, mol2_path)

    # Fix moleculetype and residue names: Sobtop uses the mol2 molecule name and
    # "MOL" as residue name, but downstream expects the resid (e.g. UNL) everywhere
    with open(itp_path) as f:
        itp_data = f.read()
    itp_data = re.sub(
        r'(\[ moleculetype \]\n;[^\n]*\n)' + re.escape(molid) + r'(\s+)',
        rf'\g<1>{resid}\g<2>', itp_data, count=1)
    itp_data = itp_data.replace(' MOL ', f' {resid} ')
    with open(itp_path, 'w') as f:
        f.write(itp_data)

    # Fix residue name in gro: Sobtop always writes "MOL", but downstream
    # expects the resid (e.g. UNL) for index group naming
    gro_path = os.path.join(wdir_ligand_cur, f'{molid}.gro')
    with open(gro_path) as f:
        gro_lines = f.readlines()
    for i in range(2, len(gro_lines) - 1):
        if len(gro_lines[i]) > 10 and 'MOL' in gro_lines[i][5:10]:
            gro_lines[i] = gro_lines[i][:5] + f'{resid:<5s}' + gro_lines[i][10:]
    with open(gro_path, 'w') as f:
        f.writelines(gro_lines)

    # Verify outputs
    for f in [itp_path, posre_path, gro_path]:
        if not os.path.isfile(f):
            logging.error(f'Expected file not found after Sobtop run: {f}')
            return None

    with open(os.path.join(wdir_ligand_cur, 'resid.txt'), 'w') as out:
        out.write(f'{molid}\t{resid}\n')

    return wdir_ligand_cur


def prep_ligand(mol_tuple, script_path, project_dir, wdir_ligand,
                conda_env_path, bash_log, gaussian_exe=None,  no_dr=False,
                activate_gaussian=None, gaussian_basis='B3LYP/6-31G*', gaussian_memory='60GB', ncpu=1,
                mol2_file=None, env=None,
                ligand_backend='ambertools', sobtop_dir=None, ligand_charge_method='gasteiger'):
    """Prepare force-field parameters and conformers for a single ligand."""
    mol, molid, resid = mol_tuple

    wdir_ligand_cur = os.path.abspath(os.path.join(wdir_ligand, molid))
    os.makedirs(wdir_ligand_cur, exist_ok=True)

    if os.path.isfile(os.path.join(wdir_ligand_cur, f'{molid}.itp')) and os.path.isfile(
            os.path.join(wdir_ligand_cur, f'posre_{molid}.itp')):
        logging.warning(
            f'{molid}.itp and posre_{molid}.itp files already exist. '
            f'Mol preparation step will be skipped for such molecule')
        if not os.path.isfile(os.path.join(wdir_ligand_cur, 'resid.txt')):
            with open(os.path.join(wdir_ligand_cur, 'resid.txt'), 'w') as out:
                out.write(f'{molid}\t{resid}\n')

        return wdir_ligand_cur

    if ligand_backend == 'sobtop':
        return prep_ligand_sobtop(
            mol_tuple, script_path=script_path, wdir_ligand=wdir_ligand,
            bash_log=bash_log, sobtop_dir=sobtop_dir,
            charge_method=ligand_charge_method, ncpu=ncpu, no_dr=no_dr,
            mol2_file=mol2_file, env=env)

    if not mol2_file or not os.path.isfile(mol2_file):
        mol2_file = os.path.join(wdir_ligand_cur, f'{molid}.mol2')
        mol_file = os.path.join(wdir_ligand_cur, f'{molid}.mol')
        # if addH:
        mol = Chem.AddHs(mol, addCoords=True)
        # reorder hydrogens for Gromacs GPU update functionality
        mol = reorder_hydrogens(mol)

        Chem.MolToMolFile(mol, mol_file)

        charge = rdmolops.GetFormalCharge(mol)

        # generate mol2
        if not os.path.isfile(mol2_file):
            # boron atom
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[#5]")):
                if gaussian_exe:
                    for file in glob(os.path.join(script_path, 'com', '*.com')):
                        prepare_gaussian_files(file_template=file,
                                               file_out=os.path.join(wdir_ligand_cur, os.path.basename(file)),
                                               ncpu=ncpu,
                                               gaussian_basis=gaussian_basis, gaussian_memory=gaussian_memory)
                    cmd = f'script_path={script_path} lfile={mol_file} input_dirname={wdir_ligand_cur} ' \
                          f'resid={resid} molid={molid} charge={charge} gaussian_version={gaussian_exe} ' \
                          f'activate_gaussian="{activate_gaussian if activate_gaussian else ""}" ' \
                          f'bash {os.path.join(script_path, "script_sh", "ligand_mol2prep_by_gaussian.sh")} ' \
                          f' >> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1'
                    if not run_check_subprocess(cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env):
                        return None
                else:
                    return None
            else:
                cmd = f'script_path={script_path} lfile={mol_file} input_dirname={wdir_ligand_cur} ' \
                      f'resid={resid} molid={molid} charge={charge} dr=yes bash {os.path.join(script_path, "script_sh", "ligand_mol2prep.sh")} ' \
                      f' >> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1',
                if not run_check_subprocess(cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env,
                                            ignore_error=True if no_dr else False):
                    if not no_dr:
                        return None
                    else:
                        logging.warning(f'Ambertools structure checking returned an error for the {mol_file} file.'
                                        f'Check the input structure carefully. Continue with -dr no mode.')
                        cmd = f'script_path={script_path} lfile={mol_file} input_dirname={wdir_ligand_cur} ' \
                          f'resid={resid} molid={molid} charge={charge} dr=no bash {os.path.join(script_path, "script_sh","ligand_mol2prep.sh")} ' \
                          f' >> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1',
                        if not run_check_subprocess(cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env):
                            return None
    else:
        mol2 = pmd.load_file(mol2_file).to_structure()
        mol2.residues[0].name = 'UNL'
        mol2.save(os.path.join(wdir_ligand_cur, f'{molid}.mol2'))
        logging.info(f'No mol2 file will be generated. {mol2_file} will be used instead')

    prepare_tleap(os.path.join(script_path, 'tleap.in'), tleap=os.path.join(wdir_ligand_cur, 'tleap.in'),
                  molid=molid, conda_env_path=conda_env_path)
    cmd = f'script_path={script_path} input_dirname={wdir_ligand_cur} ' \
          f'molid={molid} bash {os.path.join(script_path, "script_sh","ligand_prep.sh")} ' \
          f' >> {os.path.join(wdir_ligand_cur, bash_log)} 2>&1'
    if not run_check_subprocess(cmd, molid, log=os.path.join(wdir_ligand_cur, bash_log), env=env):
        return None

    # create log for molid resid corresponding
    with open(os.path.join(wdir_ligand_cur, 'resid.txt'), 'w') as out:
        out.write(f'{molid}\t{resid}\n')
    return wdir_ligand_cur


def prepare_input_ligands(ligand_fname, preset_resid, protein_resid_set, script_path, project_dir, wdir_ligand,
                          no_dr, gaussian_exe, activate_gaussian, gaussian_basis, gaussian_memory,
                          hostfile, ncpu, bash_log,
                          ligand_backend='ambertools', sobtop_dir=None, ligand_charge_method='gasteiger'):
    """Prepare parameterization inputs for multiple ligands.
     :param ligand_fname:
    :param preset_resid:
    :param script_path:
    :param project_dir:
    :param wdir_ligand:
    :param no_dr:
    :param gaussian_exe: str or None
    :param activate_gaussian: str or None
    :param hostfile:
    :param ncpu:
    :param bash_log:
    :return:
    """
    lig_wdirs = []

    if ligand_fname.endswith('.mol2'):
        mol_tuple = next(supply_mols_tuple(ligand_fname, preset_resid=preset_resid, protein_resid_set=protein_resid_set))
        res = prep_ligand(mol_tuple=mol_tuple, script_path=script_path,
            project_dir=project_dir, wdir_ligand=wdir_ligand,
            conda_env_path=os.environ["CONDA_PREFIX"],
            ncpu = ncpu, mol2_file = ligand_fname,
            bash_log=bash_log, env = os.environ.copy(),
            ligand_backend=ligand_backend, sobtop_dir=sobtop_dir,
            ligand_charge_method=ligand_charge_method)

        if res:
            lig_wdirs.append(res)

    else:
        standard_mols, boron_containing_mols = [], []
        for mol_tuple in supply_mols_tuple(ligand_fname, preset_resid=preset_resid, protein_resid_set=protein_resid_set):
            mol = mol_tuple[0]
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[#5]")):
                boron_containing_mols.append(mol_tuple)
            else:
                standard_mols.append(mol_tuple)

        # Sobtop backend handles all elements including boron without Gaussian
        if ligand_backend == 'sobtop' and boron_containing_mols:
            standard_mols.extend(boron_containing_mols)
            boron_containing_mols = []

        dask_client, cluster = None, None
        # prepare boron-containig mols
        if boron_containing_mols:
            if gaussian_exe:
                try:
                    dask_client, cluster = init_dask_cluster(hostfile=hostfile,
                                                             n_tasks_per_node=1,
                                                             use_multi_servers=True if len(boron_containing_mols) > 1 else False,
                                                             ncpu=ncpu)
                    for res in calc_dask(prep_ligand, boron_containing_mols, dask_client,
                                         script_path=script_path, project_dir=project_dir,
                                         wdir_ligand=wdir_ligand, conda_env_path=os.environ["CONDA_PREFIX"],
                                         gaussian_exe=gaussian_exe, activate_gaussian=activate_gaussian,
                                         gaussian_basis=gaussian_basis, gaussian_memory=gaussian_memory,
                                         ncpu=ncpu, bash_log=bash_log, no_dr=no_dr,
                                         env=os.environ.copy(),
                                         ligand_backend=ligand_backend, sobtop_dir=sobtop_dir,
                                         ligand_charge_method=ligand_charge_method):
                        if res:
                            lig_wdirs.append(res)
                finally:
                    if dask_client:
                        dask_client.shutdown()
                    if cluster:
                        cluster.close()
            else:
                logging.warning(
                    f'There are molecules from {ligand_fname} which have Boron atom and to prepare such molecules you need to set up Gaussian.'
                    f' Please restart the run again and use --gaussian_exe arguments')

        if standard_mols:
            try:
                dask_client, cluster = init_dask_cluster(hostfile=hostfile,
                                                         n_tasks_per_node=min(ncpu, len(standard_mols)),
                                                         use_multi_servers= True if len(standard_mols) > ncpu else False,
                                                         ncpu=ncpu)
                for res in calc_dask(prep_ligand, standard_mols, dask_client,
                                     script_path=script_path, project_dir=project_dir,
                                     wdir_ligand=wdir_ligand, no_dr=no_dr,
                                     conda_env_path=os.environ["CONDA_PREFIX"],
                                     ncpu=ncpu, bash_log=bash_log,
                                     env=os.environ.copy(),
                                     ligand_backend=ligand_backend, sobtop_dir=sobtop_dir,
                                     ligand_charge_method=ligand_charge_method):
                    if res:
                        lig_wdirs.append(res)
            finally:
                if dask_client:
                    dask_client.shutdown()
                if cluster:
                    cluster.close()

    return lig_wdirs
