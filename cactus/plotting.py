from scipy import stats
from cactus.utils import get_param_or_default
from cactus.data import list_directories, list_files, load_json
from cactus.constants import *
import matplotlib.pyplot as plot
import numpy

def bootstrap(data, n_boot=10000, ci=95):
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = numpy.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(numpy.mean(sample, axis=0))
    b = numpy.array(boot_dist)
    s1 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def tsplot(data, params, alpha=0.12, **kw):
    data = numpy.array(data)
    default_x = list(range(data.shape[1]))
    x = get_param_or_default(params, "x_axis_values", default_x)[:len(default_x)]
    est = numpy.mean(data, axis=0)
    ci = get_param_or_default(params, "ci", 95)
    cis = bootstrap(data, ci=ci)
    color = get_param_or_default(params, "color", None)
    label = params["label"]
    x_label = params["x_label"]
    y_label = params["y_label"]
    plot.title(get_param_or_default(params, PLOT_TITLE, ""))
    if color is not None:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
        handle = plot.plot(x,est,label=label,color=color,**kw)
    else:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, **kw)
        handle = plot.plot(x,est,label=label, **kw)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.margins(x=0)
    return handle

def plot_runs(params):
    data = []
    directory_count = 0
    filter_size = get_param_or_default(params, "filter_size", None)
    stats_label = get_param_or_default(params, STATS_LABEL, SUCCESS_RATE)
    use_runtime = get_param_or_default(params, "use_runtime", False)
    path = params[DIRECTORY]
    filename = get_param_or_default(params, "filename", "results.json")
    data_prefix_pattern = params[DATA_PREFIX_PATTERN]
    json_data = None
    x = None
    for directory in list_directories(path, lambda x: x.startswith(data_prefix_pattern)):
        for json_file in list_files(directory, lambda x: x == filename):
            json_data = load_json(json_file)
            values = json_data[stats_label]
            if filter_size is not None and filter_size > 1:
                kernel = numpy.ones(filter_size)/filter_size
                values = numpy.convolve(values, kernel, mode='valid')
            data.append(values)
            directory_count += 1
            if use_runtime:
                x = json_data["training_time"]
    if len(data) > 0:
        if "x_axis_values" not in params and x is not None:
            params["x_axis_values"] = numpy.cumsum(x)/3600
        print(data_prefix_pattern, "{} runs".format(directory_count))
        result = tsplot(data, params)
        params.pop("x_axis_values")
        return result
    return json_data

def figure_size(size):
    plot.figure(figsize=size)

def x_limit(lim):
    plot.xlim(lim)

def y_limit(lim):
    plot.ylim(lim)

def show(showgrid=True, legend_position=None):
    if showgrid:
        plot.grid()
    if legend_position is not None:
        plot.legend(loc=legend_position)
    else:
        plot.legend()
    plot.show()