#!/bin/bash
#  args: wdir index_protein_ligand tu dtstep
cd $wdir

gmx trjconv -s md_out.tpr -f md_out.xtc -pbc nojump -o md_out_noj_noPBC.xtc <<< "System" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }
#gmx trjconv -s md_out.tpr -f md_out.xtc -o md_out_noPBC.xtc -pbc mol -center <<< "Protein  System"
gmx trjconv -s md_out.tpr -f md_out_noj_noPBC.xtc -o md_centermolsnoPBC.xtc -pbc mol -center -n index.ndx  <<< "$index_protein_ligand  System" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }
# use it for PBSA https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA/issues/33
gmx trjconv -s md_out.tpr -f md_centermolsnoPBC.xtc -fit rot+trans -o md_fit.xtc -n index.ndx <<< "$index_protein_ligand  System" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }

gmx trjconv -s md_out.tpr -f md_fit.xtc -dt $dtstep -o md_short_forcheck.xtc <<< "System" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }

gmx rms -s md_out.tpr -f md_fit.xtc -o rmsd.xvg -n index.ndx -tu $tu <<< "Backbone  Backbone" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }
gmx rms -s em.tpr -f md_fit.xtc -o rmsd_xtal.xvg -n index.ndx -tu $tu <<< "Backbone  Backbone" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }

gmx gyrate -s md_out.tpr -f md_fit.xtc -n index.ndx -o gyrate.xvg <<< "Protein" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }
gmx rmsf -s md_out.tpr -f md_fit.xtc -n index.ndx -o rmsf.xvg -oq rmsf.pdb -res <<< "Protein" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }

gmx trjconv -s md_out.tpr -f md_fit.xtc -o frame.pdb -b 10 -e 11  -n index.ndx <<< "System" || { >&2 echo "Failed to run command  at line ${LINENO} of ${BASH_SOURCE}" && exit 1; }
