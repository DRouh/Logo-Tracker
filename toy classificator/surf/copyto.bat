@echo off
set src_folder=e:\1234\surfall
set dst_folder=e:\1234\surftrain
set file_list=e:\1234\390filesforsiftKMeans.txt

if not exist "%dst_folder%" mkdir "%dst_folder%"

for /f "delims=" %%f in (%file_list%) do (
    xcopy "%src_folder%\%%f" "%dst_folder%\"
)