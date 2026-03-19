using YAML

function load_config(path::String)
    return YAML.load_file(path)
end