#!/bin/bash
#  args: input_dirname molid sobtop_dir nthreads
#  Generate GROMACS topology (.itp), coordinate (.gro) and position restraints (posre_*.itp)
#  from a ligand .mol2 file using Sobtop with prebuilt GAFF parameters.
cd "$input_dirname" || { echo "Failed to cd to $input_dirname at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }

# Copy required Sobtop data files to working directory
for f in sobtop.ini atomtype ATOMTYPE_GFF.DEF ATOMTYPE_AMBER.DEF CORR_NAME_TYPE.DAT LJ_param.dat assign_AT.dat bondcrit.dat bonded_param.dat; do
    cp "$sobtop_dir/$f" . || { echo "Failed to copy $f at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }
done

chmod +x atomtype || { echo "Failed to chmod atomtype at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }

# Ensure single-threaded execution
sed -i 's/^nthreads=.*/nthreads=1/' sobtop.ini
export OMP_NUM_THREADS=${nthreads:-1}

# Run Sobtop non-interactively via printf pipe (verified 9-step sequence)
# mol2 -> 1(topology) -> 2(GAFF+UFF, auto-advances) -> 4(prebuilt+guess) -> top -> itp -> 2(gro) -> gro -> 0(exit)
printf '%s\n' "$molid.mol2" "1" "2" "4" "$molid.top" "$molid.itp" "2" "$molid.gro" "0" \
    | "$sobtop_dir/sobtop" || { echo "Failed to run sobtop at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }

# Verify outputs exist
if [ ! -s "$molid.itp" ] || [ ! -s "$molid.gro" ]; then
    echo "Sobtop did not generate expected output files at line ${LINENO} in ${BASH_SOURCE}"
    exit 1
fi

# Generate position restraints (heavy atoms only)
gmx make_ndx -f "$molid.gro" -o index.ndx << INPUT || { echo "Failed to run make_ndx at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }
2 & ! a H*
q
INPUT
printf '%s\n' "3" | gmx genrestr -f "$molid.gro" -o "posre_$molid.itp" -n index.ndx -fc 1000 1000 1000 || { echo "Failed to run genrestr at line ${LINENO} in ${BASH_SOURCE}" && exit 1; }
