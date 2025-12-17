# Medial Python Binding

## Relase Notes - 1.1.0
* Split medpython-etl into a different package, medpython will set medpython-etl that it will be installed
* Added functionality for med.Model - apply_model_changes, add_post_processors_json_string_to_model

## Relase Notes - 1.0.6
* Fix code to compile in ARM computer
* Removed ipython dependency
* Readme fixes
* Removed the Boost libraries dependency, using only headers. Simpler compile process

## Relase Notes - 1.0.5
* Adjustments to compile with Alpine/Musl library
* Bugfix issue #5
* Fix code to support and compile in osx

## Relase Notes - 1.0.4
* Fixed all issues to compile also in windows

## Relase Notes - 1.0.3
* Updated documentation

## Release Notes - 1.0.2
* First version deployed on PyPi with CI/CD of github
* Added MedConvert to med library to allow loading of data into PidRepository data format without external tools like `Flow`
* Added compilation ability based on shared library of Boost for self compliation ability

## Release Notes - 1.0.1
* First version - depolying current status
