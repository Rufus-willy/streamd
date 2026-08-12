"""蛋白 PDB 缺失重原子修复工具。

使用 pdbfixer 补全缺失的重原子（侧链/骨架），不加氢、不建模缺失 loop，
保持与 gmx pdb2gmx -ignh 流程兼容。
"""
import logging
import os


def fix_missing_atoms(pdb_in, pdb_out):
    """检测并补全 PDB 中的缺失重原子。

    Parameters
    ----------
    pdb_in : str
        输入 PDB 文件路径
    pdb_out : str
        修复后的 PDB 输出路径

    Returns
    -------
    tuple
        (was_fixed: bool, missing_residues_info: list of str)
        was_fixed 表示是否进行了修复，missing_residues_info 是被修复的残基描述列表
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        raise ImportError(
            'pdbfixer 未安装，无法自动修复缺失原子。'
            '请安装: pip install pdbfixer'
            '或使用 --no_fix_protein 跳过修复。')

    fixer = PDBFixer(filename=pdb_in)
    # 不建模缺失的 loop/残基
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    # 替换非标准残基
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    # 检测缺失原子
    fixer.findMissingAtoms()
    missing_info = []
    for residue_key, atoms in fixer.missingAtoms.items():
        resname = residue_key.name
        resnum = residue_key.id
        chain_id = residue_key.chain.id
        atom_names = [a.name for a in atoms]
        missing_info.append(f'{resname}{resnum}({chain_id}): {", ".join(atom_names)}')
    if not missing_info:
        # 无缺失原子，直接复制（或不处理）
        return False, []
    logging.info(f'检测到缺失原子，正在用 pdbfixer 修复: {"; ".join(missing_info)}')
    fixer.addMissingAtoms(seed=42)
    # 不加氢，保持 -ignh 流程
    with open(pdb_out, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    n_heavy = sum(1 for a in fixer.topology.atoms() if a.element.symbol != 'H')
    logging.info(f'蛋白修复完成，输出 {n_heavy} 个重原子到 {pdb_out}')
    return True, missing_info
