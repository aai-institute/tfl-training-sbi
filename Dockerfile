FROM jupyter/minimal-notebook:python-3.10.11


USER root
RUN apt-get update && apt-get upgrade -y

# pandoc needed for docs, see https://nbsphinx.readthedocs.io/en/0.7.1/installation.html?highlight=pandoc#pandoc
# gh-pages action uses rsync
RUN apt-get -y install pandoc git-lfs rsync gcc python3-dev

USER ${NB_UID}

WORKDIR /tmp
# COPY build_scripts build_scripts
# RUN bash build_scripts/install_presentation_requirements.sh

# COPY requirements-test.txt .
# RUN pip install -r requirements-test.txt


# NOTE: this breaks down when requirements contain pytorch (file system too large to fit in RAM, even with 16GB)
# NOTE: this might break down when requirements contain pytorch (file system too large to fit in RAM, even with 16GB)
# If pytorch is a requirement, the suggested solution is to keep a requirements-docker.txt and only install
# the lighter requirements. The install of the remaining requirements then has to happen at runtime
# instead of build time (usually as part of the entrypoint)
# COPY requirements.txt .
# RUN pip install -r requirements.txt


# Start of HACK: the home directory is overwritten by a mount when a jhub server is started off this image
# Thus, we create a jovyan-owned directory to which we copy the code and then move it to the home dir as part
# of the entrypoint
ENV CODE_DIR=/tmp/code

# RUN mkdir $CODE_DIR

# COPY --chown=${NB_UID}:${NB_GID} entrypoint.sh $CODE_DIR

# RUN chmod +x "${CODE_DIR}/"entrypoint.sh
# Unfortunately, we cannot use ${CODE_DIR} in the ENTRYPOINT directive, so we have to hardcode it
# Keep in sync with the value of CODE_DIR above
# ENTRYPOINT ["/tmp/code/entrypoint.sh"]

# End of HACK

# WORKDIR "${HOME}"

# create a directory for the code and copy the code into it
# in the entry point, the contents of this directory will be moved to the home
# directory in order to mitigate the issue of the home directory being
# overwritten by the mount when a jhub server is started off of this image
RUN mkdir $CODE_DIR
COPY --chown=${NB_UID}:${NB_GID} . $CODE_DIR
RUN chmod +x "${CODE_DIR}/"entrypoint.sh

# set the working directory to the code directory and copy over all files
# install all presentation dependencies as well as the code requirements
# themselves
WORKDIR $CODE_DIR
RUN bash build_scripts/install_presentation_requirements.sh
RUN pip install -e "."

# finally, set the entrypoint which copies the code to the home directory and
# trusts all notebooks 
ENTRYPOINT ["/tmp/code/entrypoint.sh"]
