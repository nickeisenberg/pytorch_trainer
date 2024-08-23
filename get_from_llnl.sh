rsync -av \
    --exclude='.git' \
    --exclude='*.pth' \
    --exclude='*.png' \
    --exclude='build' \
    --exclude='*.egg-info' \
    eisenbnt@pascal.llnl.gov:/p/lustre1/nova/software/pytorch_trainer \
    /home/nicholas/GitRepos/
