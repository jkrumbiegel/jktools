import numpy as np
import typing


class Animation:

    def __init__(
            self,
            name: str,
            values: np.ndarray,
            durations: np.ndarray,
            easings: typing.Union[typing.Union[str, typing.Callable], typing.List[typing.Union[str, typing.Callable]]],
            debug=False
    ):
        self._name = name
        self._values = values
        self._durations = durations
        self._debug = debug

        self._n_intervals = durations.size
        self._timepoints = np.concatenate((np.zeros(1), np.cumsum(durations)))

        if self._n_intervals + 1 != self._values.shape[0]:
            raise(Exception(f"The number of values ({self._values.shape[0]}) is not one more than"
                            f" the number of interval durations ({self._n_intervals})."))

        if isinstance(easings, str):
            self._easings = [easings for _ in range(self._n_intervals)]
        elif isinstance(easings, list):
            if len(easings) != self._n_intervals:
                raise(Exception(f"The number of easings ({len(easings)}) is not equal "
                                f"to the number of intervals ({self._n_intervals})."))

    def update(
            self,
            variables: typing.Dict[str, np.ndarray],
            t: float
    ):
        active_interval = get_active_interval(t, self._timepoints)

        self._dprint("t:", t, "\tinterval:", active_interval + 1, "of", self._n_intervals)
        if active_interval is None or active_interval >= self._n_intervals:
            return

        interval_fraction = (t - self._timepoints[active_interval]) / self._durations[active_interval]
        self._dprint("interval fraction:", interval_fraction)

        eased_fraction = self._calculate_easing(interval_fraction, self._easings[active_interval])

        start_value = self._values[active_interval, ...]
        end_value = self._values[active_interval + 1, ...]

        interpolated_value = start_value + eased_fraction * (end_value - start_value)

        self._dprint(interpolated_value)

        variables[self._name] = interpolated_value

    def _calculate_easing(self, fraction, easing):

        if callable(easing):
            return easing(fraction)

        elif easing == "linear":
            return fraction

        elif easing == "sin_io":
            return (np.sin(fraction * np.pi - (np.pi / 2)) + 1) / 2

        elif easing == "quad_in":
            return fraction ** 2

        else:
            raise(Exception(f"Easing ({easing}) neither callable, nor recognized type."))

    def _dprint(self, *args):
        if self._debug:
            print(*args)


def get_active_interval(t: float, timepoints: np.ndarray) -> typing.Optional[int]:

        if t > timepoints.max():
            # time is after last interval
            return timepoints.size - 1

        differences = timepoints - t

        # past values are negative, upcoming positive
        # we look for the smallest negative or zero

        differences[differences > 0] = np.nan

        if np.all(np.isnan(differences)):
            return None

        closest_index = np.nanargmax(differences)

        return closest_index
