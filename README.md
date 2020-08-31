# Numerical Optimization

Numerical Optimization is a C++ library for executing and testing different unconstrained optimisation algorithms. The project is based on [Vilin](https://github.com/markomil/vilin-numerical-optimization) written in MATLAB. Although this library is standalone, the idea is to eventually delegate heavy Vilin computation directly to C++ to speed it up.

## Setting up

Additional libraries needed to build this project:
- `Armadillo` - more information [here](http://arma.sourceforge.net)
- `OpenBLAS` - more information [here](https://www.openblas.net/)

Optionally you can use alternative backend library (other than `OpenBLAS`) like Intel's `MKL`. For more information about this see notes on `Armadillo` site.

### Linux

On most distributions, you can install dependencies through package manager.

#### Debian/Ubuntu

```bash
sudo apt install build-essential cmake libarmadillo-dev
```

#### Fedora

```bash
sudo dnf install @development-tools cmake armadillo-devel
```

To set up environment run (in project directory):

```bash
mkdir build
cd build
cmake ..
```

Available targets:
- `numopt` - library
- `benchmark` - benchmark for `numopt` library
- `test` - executable where you can see how to call functions and test results

To build all targets run (in `build` directory):
```bash
make
```

To build specific target run (in `build` directory):
```bash
make <target_name>
```

### Windows

We will describe steps to set up `Visual Studio` project. Other CMake targets are also available.

1. Install [CMake](https://cmake.org/download/) for Windows
1. Install Visual Studio C++ tools (including MSVC v142 C++ build tools)
1. Download `Armadillo` archive from [here](http://arma.sourceforge.net/download.html) and extract its content to
`C:\Program files\Armadillo`
1. Make `lib_win64` directory in `C:\Program files\Armadillo`
1. Download `OpenBLAS` from [here](https://github.com/xianyi/OpenBLAS/releases) (compiled library, not source code) and copy `lib\libopenblas.lib` to `C:\Program files\Armadillo\lib_win64`
1. Open `cmake-gui` and set your source and build directory to location where you cloned this repository
1. Run Configure and enter your Visual Studio version\
**Note**: Here you can configure targets other than Visual Studio
1. Run Generate
1. Now you can open generated Visual Studio solution.\
For some reason Visual Studio sometimes doesn't see additional include and library directories listed in CMake script. In that case these steps should be done as well.
1. Do this for every project (except meta ones like 'ALL_BUILD' and 'ZERO_CHECK')\
Right click on project and open Properties\
**Note**: Make sure you set up this for all configurations.
   - under `C/C++ -> General -> Additional Include Directories` add `C:\Program files\Armadillo\include`
   - under `Linker -> General -> Additional Library Directories` add `C:\Program files\Armadillo\lib_win64`
   - under `Linker -> Input -> Additional Dependencies` add `libopenblas.lib`
1. For every executable, copy `bin\libopenblas.dll` from `OpenBLAS` archive to folder where that executable is built.
1. (Optional) Manually recreate `numopt` project directory structure in Visual Studio with filters and adding existing files.

**Note:** Any project modifications should be done through CMake. Visual Studio solution and other generated files are ignored by `git`.

## Optimization Methods

- Gradient
  - Gradient Descent
  - Momentum
- Conjugate Gradient
  - Fletcher-Reeves
  - Polak-Ribiere
  - Hestenes-Stiefel
  - Dai-Yuan
  - CG Descent
- Quasi Newton
  - SR1
  - DFP
  - BFGS
  - L-BFGS

## Line Search methods

- Fixed Step Size
- Binary
- Armijo
- Goldstein
- Wolfe
- Strong Wolfe
- Approx. Wolfe

## Future steps

- add missing methods and functions
- make python bindings
  - simpler interface to library, easier to call methods with python code, rather than compiling cpp program
  - it may be easier to use [xtensor](https://github.com/xtensor-stack/xtensor) instead of armadillo for integration with numpy
    - benchmark performance between armadillo and xtensor
- GUI desktop application (cross platform)
  - mimic vilin GUI functionalities
  - [electron](https://www.electronjs.org/) app
    - good choice because it provides cross platform app using web technologies
    - has mature and interactive graphing and visualization tools ([chartjs](https://www.chartjs.org/))
    - try writing it in Typescript to avoid pain :smiley:
  - python
    - integration with matplotlib and above described python bindings
    - [Tkinter](https://docs.python.org/3/library/tkinter.html)
    - PyQT

## Authors

* **Ivan Stosic** - ivan100sic@gmail.com
* **Igor Stosic** - igor.stosic@pmf.edu.rs
* **Lazar Stamenkovic** - lazar.stamenkovic@pmf.edu.rs
* **Nenad Todorovic** - nenad.todorovic@pmf.edu.rs
* **Danilo Petkovic** - danilo.petkovic@pmf.edu.rs
