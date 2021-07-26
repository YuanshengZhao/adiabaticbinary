powershell "python %1 %2 %3 | tee (\"%1\"+\"-\"+$(Get-Date -Format \"yyyy-MM-dd-HH-mm\")+\".log\")"
