$pdf_mode = 1;
$interaction = 'nonstopmode';
$synctex = 1;
$out_dir = 'build';
$pdflatex = 'pdflatex -shell-escape %O %S';

# copy PDF to project root after build
$success_cmd = 'cp %D %R.pdf';
