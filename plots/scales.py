import math
from matplotlib import cbook
import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import (
    AutoLocator,
    FixedLocator,
    Formatter,
    FuncFormatter,
    Locator,
    LogFormatterSciNotation,
    LogLocator,
)


class LogShiftScale(mscale.ScaleBase):
    """
    Scales data in range [0, 1) using a log scale in the vicinity of 1.

    The scale function:
      -ln(1 - x)

    The inverse scale function:
      1 - exp(-y)

    Since the LogOne scale tends to infinity near 1,
    there is user-defined threshold, above which nothing
    will be plotted.
    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``ax.set_yscale("logShift")`` would be
    # used to select this scale.
    name = "logShift"

    def __init__(
        self, axis, *, base=10, thresh=None, subs=None, shift: float = 1.0, **kwargs
    ):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)
        if thresh is not None and thresh >= 1.0:
            raise ValueError("thresh must be less than 1.0")
        self.thresh = thresh
        self.base = base
        self.subs = subs
        self.shift = shift

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.LogShiftTransform(self.thresh, self.shift)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(lambda x, pos=None: f"{np.degrees(x):.0f}\N{DEGREE SIGN}")

        axis.set_major_locator(LogShiftLocator(self.base))
        axis.set_major_formatter(LogShiftFormatterSciNotation(self.base))
        axis.set_minor_locator(LogShiftLocator(self.base, self.subs))
        axis.set_minor_formatter(
            LogShiftFormatterSciNotation(
                self.base, labelOnlyBase=(self.subs is not None)
            )
        )

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return vmin, min(vmax, self.thresh if self.thresh is not None else self.shift)

    class LogShiftTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self, thresh, shift=1.0):
            mtransforms.Transform.__init__(self)
            self.shift = shift
            self.thresh = thresh if thresh is not None else self.shift

        def transform_non_affine(self, a):
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the Mercator scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            masked = ma.masked_where((a >= self.thresh), a)
            if masked.mask.any():
                return -ma.log(self.shift - masked)
            else:
                return -np.log(self.shift - a)

        def inverted(self):
            """
            Override this method so Matplotlib knows how to get the
            inverse transform for this transform.
            """
            return LogShiftScale.InverseLogShiftTransform(self.thresh, self.shift)

    class InverseLogShiftTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh, shift: float = 1.0):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.shift = shift

        def transform_non_affine(self, a):
            return self.shift - ma.exp(-a)

        def inverted(self):
            return LogShiftScale.LogShiftTransform(self.thresh, self.shift)


# =============================================================================
# Shifted log locator
# =============================================================================


class LogShiftLocator(Locator):
    """
    Place logarithmically spaced ticks.

    Places ticks at the values ``subs[j] * base**i``.
    """

    def __init__(self, base=10.0, subs=(1.0,), numticks=None, shift=1.0):
        """
        Parameters
        ----------
        base : float, default: 10.0
            The base of the log used, so major ticks are placed at ``base**n``, where
            ``n`` is an integer.
        subs : None or {'auto', 'all'} or sequence of float, default: (1.0,)
            Gives the multiples of integer powers of the base at which to place ticks.
            The default of ``(1.0, )`` places ticks only at integer powers of the base.
            Permitted string values are ``'auto'`` and ``'all'``. Both of these use an
            algorithm based on the axis view limits to determine whether and how to put
            ticks between integer powers of the base:
            - ``'auto'``: Ticks are placed only between integer powers.
            - ``'all'``: Ticks are placed between *and* at integer powers.
            - ``None``: Equivalent to ``'auto'``.
        numticks : None or int, default: None
            The maximum number of ticks to allow on a given axis. The default of
            ``None`` will try to choose intelligently as long as this Locator has
            already been assigned to an axis using `~.axis.Axis.get_tick_space`, but
            otherwise falls back to 9.
        shift : float, default: 1.0
            The value which should be considered as zero on a standard log scale.
        """
        if numticks is None:
            numticks = "auto"
        self._base = float(base)
        self._set_subs(subs)
        self.numticks = numticks
        self.shift = shift  # The value at which the axis is considered singular

    def set_params(self, base=None, subs=None, numticks=None, shift=None):
        """Set parameters within this locator."""
        if base is not None:
            self._base = float(base)
        if subs is not None:
            self._set_subs(subs)
        if numticks is not None:
            self.numticks = numticks
        if shift is not None:
            self.shift = shift

    def _set_subs(self, subs):
        """
        Set the minor ticks for the log scaling every ``base**i*subs[j]``.
        """
        if subs is None:  # consistency with previous bad API
            self._subs = "auto"
        elif isinstance(subs, str):
            self._subs = subs
        else:
            try:
                self._subs = np.asarray(subs, dtype=float)
            except ValueError as e:
                raise ValueError(
                    "subs must be None, 'all', 'auto' or "
                    "a sequence of floats, not "
                    f"{subs}."
                ) from e
            if self._subs.ndim != 1:
                raise ValueError(
                    "A sequence passed to subs must be "
                    "1-dimensional, not "
                    f"{self._subs.ndim}-dimensional."
                )

    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == "auto":
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks

        b = self._base
        v0 = self.shift

        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError(
                    "Data has no positive values, and therefore cannot be log-scaled."
                )

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        log_vmin = math.log(v0 - vmax) / math.log(b)
        log_vmax = math.log(v0 - vmin) / math.log(b)

        numdec = math.ceil(log_vmax - 1e-3) - math.floor(log_vmin + 1e-3)

        if isinstance(self._subs, str):
            if numdec > 10 or b < 3:
                if self._subs == "auto":
                    return np.array([])  # no minor or major ticks
                else:
                    subs = np.array([1.0])  # major ticks
            else:
                _first = 2.0 if self._subs == "auto" else 1.0
                subs = np.arange(_first, b)
        else:
            subs = self._subs

        # Get decades between major ticks.
        stride = max(math.ceil(numdec / (numticks - 1)), 1)

        # if we have decided that the stride is as big or bigger than
        # the range, clip the stride back to the available range - 1
        # with a floor of 1.  This prevents getting axis with only 1 tick
        # visible.
        if stride >= numdec:
            stride = max(1, numdec - 1)

        # Does subs include anything other than 1?  Essentially a hack to know
        # whether we're a major or a minor locator.
        have_subs = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)

        decades = np.arange(
            math.floor(log_vmin) - stride, math.ceil(log_vmax) + 2 * stride, stride
        )

        if have_subs:
            if stride == 1:
                ticklocs = np.concatenate(
                    [v0 - subs * decade_start for decade_start in b**decades]
                )
            else:
                ticklocs = np.array([])
        else:
            ticklocs = v0 - b**decades

        if (
            len(subs) > 1
            and stride == 1
            and ((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1
        ):
            # If we're a minor locator *that expects at least two ticks per
            # decade* and the major locator stride is 1 and there's no more
            # than one minor tick, switch to AutoLocator.
            return AutoLocator().tick_values(vmin, vmax)
        else:
            return self.raise_if_exceeds(ticklocs)

    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        b = self._base

        vmin, vmax = self.nonsingular(vmin, vmax)

        if mpl.rcParams["axes.autolimit_mode"] == "round_numbers":
            vmin = _decade_less(vmin, b)
            vmax = _decade_greater(vmax, b)

        return vmin, vmax

    def nonsingular(self, vmin, vmax):
        v0 = self.shift

        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = v0 - 1, v0 - 0.1  # Initial range, no data plotted yet.
        elif vmax >= v0:
            _api.warn_external(
                "Data has no positive values, and therefore cannot be " "log-scaled."
            )
            vmin, vmax = v0 - 1, v0 - 0.1  # No data plotted.
        else:
            # Consider shared axises
            minpos = min(axis.get_minpos() for axis in self.axis._get_shared_axis())
            if not np.isfinite(minpos):
                minpos = 1e-300  # This should never take effect.
            if v0 - vmin <= 0:
                vmin = v0 - minpos
            if vmin == vmax:
                vmin = _decade_less(vmin, self._base, self.shift)
                vmax = _decade_greater(vmax, self._base, self.shift)
        return vmin, vmax


# =============================================================================
# Singular log formatter
# =============================================================================


class LogShiftFormatter(Formatter):
    """
    Base class for formatting ticks on a log or symlog scale.

    It may be instantiated directly, or subclassed.

    Parameters
    ----------
    base : float, default: 10.
        Base of the logarithm used in all calculations.

    shift : float, default: 1.0
        The value which should be considered as zero on a standard log scale.

    labelOnlyBase : bool, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.

    minor_thresholds : (subset, all), default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.

    linthresh : None or float, default: None
        If a symmetric log scale is in use, its ``linthresh``
        parameter must be supplied here.

    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.

    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.

    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.

    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.

    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """

    def __init__(
        self,
        base=10.0,
        shift=1.0,
        labelOnlyBase=False,
        minor_thresholds=None,
        linthresh=None,
    ):

        self.set_base(base)
        self.set_label_minor(labelOnlyBase)
        if minor_thresholds is None:
            if mpl.rcParams["_internal.classic_mode"]:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh
        self._shift = shift

    def set_base(self, base):
        """
        Change the *base* for labeling.

        .. warning::
           Should always match the base used for :class:`LogLocator`
        """
        self._base = float(base)

    def set_label_minor(self, labelOnlyBase):
        """
        Switch minor tick labeling on or off.

        Parameters
        ----------
        labelOnlyBase : bool
            If True, label ticks only at integer powers of base.
        """
        self.labelOnlyBase = labelOnlyBase

    def set_shift(self, shift):
        """
        Set the value at which the axis is considered singular.
        """
        self._shift = shift

    def set_locs(self, locs=None):
        """
        Use axis view limits to control which ticks are labeled.

        The *locs* parameter is ignored in the present algorithm.
        """
        if np.isinf(self.minor_thresholds[0]):
            self._sublabels = None
            return

        # Handle symlog case:
        linthresh = self._linthresh
        if linthresh is None:
            try:
                linthresh = self.axis.get_transform().linthresh
            except AttributeError:
                pass

        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if linthresh is None and vmin <= 0:
            # It's probably a colorbar with
            # a format kwarg setting a LogFormatter in the manner
            # that worked with 1.5.x, but that doesn't work now.
            self._sublabels = {1}  # label powers of base
            return

        b = self._base
        v0 = self._shift
        if linthresh is not None:  # symlog
            # Only compute the number of decades in the logarithmic part of the
            # axis
            numdec = 0
            if vmin < -linthresh:
                rhs = min(v0 - vmax, -linthresh)
                numdec += math.log((v0 - vmin) / rhs) / math.log(b)
            if vmax > linthresh:
                lhs = max(v0 - vmin, linthresh)
                numdec += math.log(v0 - vmax / lhs) / math.log(b)
        else:
            vmin = math.log(v0 - vmin) / math.log(b)
            vmax = math.log(v0 - vmax) / math.log(b)
            numdec = abs(vmax - vmin)

        if numdec > self.minor_thresholds[0]:
            # Label only bases
            self._sublabels = {1}
        elif numdec > self.minor_thresholds[1]:
            # Add labels between bases at log-spaced coefficients;
            # include base powers in case the locations include
            # "major" and "minor" points, as in colorbar.
            c = np.geomspace(1, b, int(b) // 2 + 1)
            self._sublabels = set(np.round(c))
            # For base 10, this yields (1, 2, 3, 4, 6, 10).
        else:
            # Label all integer multiples of base**n.
            self._sublabels = set(np.arange(1, b + 1))

    def _num_to_string(self, x, vmin, vmax):
        if x > 10000:
            s = "%1.0e" % x
        elif x < 1:
            s = "%1.0e" % x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        # docstring inherited
        if x == 0.0:  # Symlog
            return "0"

        x = abs(x)
        b = self._base
        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))

        if self.labelOnlyBase and not is_x_decade:
            return ""
        if self._sublabels is not None and coeff not in self._sublabels:
            return ""

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        s = self._num_to_string(x, vmin, vmax)
        return self.fix_minus(s)

    def format_data(self, value):
        with cbook._setattr_cm(self, labelOnlyBase=False):
            return cbook.strip_math(self.__call__(value))

    def format_data_short(self, value):
        # docstring inherited
        return ("%-12g" % value).rstrip()

    def _pprint_val(self, x, d):
        # If the number is not too big and it's an int, format it as an int.
        if abs(x) < 1e4 and x == int(x):
            return "%d" % x
        fmt = (
            "%1.3e"
            if d < 1e-2
            else (
                "%1.3f"
                if d <= 1
                else "%1.2f" if d <= 10 else "%1.1f" if d <= 1e5 else "%1.1e"
            )
        )
        s = fmt % x
        tup = s.split("e")
        if len(tup) == 2:
            mantissa = tup[0].rstrip("0").rstrip(".")
            exponent = int(tup[1])
            if exponent:
                s = "%se%d" % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip("0").rstrip(".")
        return s


# =============================================================================
# Mathtext formatter
# =============================================================================


class LogShiftFormatterMathtext(LogShiftFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _non_decade_format(self, op_string, base, shift, fx, usetex):
        """Return string for non-decade locations."""
        return r"$\mathdefault{%g%s%s^{%.2f}}$" % (shift, op_string, base, fx)

    def __call__(self, x, pos=None):
        # docstring inherited
        if x == 0:  # Symlog
            return r"$\mathdefault{0}$"

        v0 = self._shift

        sign_string = "-" if x < v0 else ""
        op_string = "-" if x < v0 else "+"
        x0 = x
        x = abs(x - v0)
        b = self._base

        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))

        if self.labelOnlyBase and not is_x_decade:
            return ""
        if self._sublabels is not None and coeff not in self._sublabels:
            return ""

        if is_x_decade:
            fx = round(fx)

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = "%d" % b
        else:
            base = "%s" % b

        if abs(fx) < 4:
            return r"$\mathdefault{%g}$" % x0
        elif not is_x_decade:
            usetex = mpl.rcParams["text.usetex"]
            return self._non_decade_format(op_string, base, v0, fx, usetex)
        else:
            return r"$\mathdefault{%g%s%s^{%d}}$" % (v0, op_string, base, fx)


# =============================================================================
# SciNotation formatter
# =============================================================================


class LogShiftFormatterSciNotation(LogShiftFormatterMathtext):
    """
    Format values following scientific notation in a logarithmic axis.
    """

    def _non_decade_format(self, op_string, base, shift, fx, usetex):
        """Return string for non-decade locations."""
        b = float(base)
        exponent = math.floor(fx)
        coeff = b ** (fx - exponent)
        if _is_close_to_int(coeff):
            coeff = round(coeff)
        return r"$\mathdefault{%g%s%g\times%s^{%d}}$" % (
            shift,
            op_string,
            coeff,
            base,
            exponent,
        )


# =============================================================================
# Helper functions
# =============================================================================


def _decade_less(x, base, shift, direction: int = -1):
    """
    Return the largest value less than x that is an integer power of base.
    """
    if direction < 0:
        if x >= shift:
            return shift
        return shift - base ** math.ceil(math.log(shift - x, base))
    else:
        if x <= shift:
            return shift
        return shift + base ** math.floor(math.log(x - shift, base))


def _decade_greater(x, base, shift, direction: int = -1):
    """
    Return the smallest value greater than x that is an integer power of base.
    """
    if direction < 0:
        if x >= shift:
            return shift
        return shift - base ** math.floor(math.log(shift - x, base))
    else:
        if x <= shift:
            return shift
        return shift + base ** math.ceil(math.log(x - shift, base))


def _is_close_to_int(x):
    return np.isclose(x, np.round(x))


mscale.register_scale(LogShiftScale)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = 1 - np.geomspace(1e-6, 1, 101)[::-1]
    s = (1 - t) ** 2

    plt.plot(t, s, "-", lw=2)
    plt.xscale("logShift")

    plt.xlabel("t")
    plt.ylabel("s")
    plt.title("Log One projection")
    plt.grid(True)
    plt.xlim(0, 1 - 1e-5)

    plt.show()
