from abc import ABC, abstractmethod


class BaseProcessing(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ApplyToDict(BaseProcessing):
    def __init__(self, proc, unpack=False):
        self._proc = proc
        self._unpack = unpack

    def __call__(self, input_dict):
        if self._unpack:
            return {key: self._proc(*value) for key, value in input_dict.items()}
        else:
            return {key: self._proc(value) for key, value in input_dict.items()}


class ApplyToList(BaseProcessing):
    def __init__(self, proc, unpack=False):
        self._proc = proc
        self._unpack = unpack

    def __call__(self, input_list):
        if self._unpack:
            return [self._proc(*item) for item in input_list]
        else:
            return [self._proc(item) for item in input_list]


class Compose(BaseProcessing):
    def __init__(self, proc_list):
        self._proc_list = proc_list

    def append(self, proc):
        self._proc_list.append(proc)

    def __call__(self, args):
        data = args
        for proc in self._proc_list:
            data = proc(data)
        return data


class Lambda(BaseProcessing):
    def __init__(self, lm):
        self._lm = lm

    def __call__(self, *args):
        return self._lm(*args)


class PrintData(BaseProcessing):
    def __init__(self):
        pass

    def __call__(self, args):
        print(args)
        return args
