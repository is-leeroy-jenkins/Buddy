var zT = Object.defineProperty;
var pE = (p) => {
  throw TypeError(p);
};
var UT = (p, t, e) => t in p ? zT(p, t, { enumerable: !0, configurable: !0, writable: !0, value: e }) : p[t] = e;
var Z = (p, t, e) => UT(p, typeof t != "symbol" ? t + "" : t, e), v0 = (p, t, e) => t.has(p) || pE("Cannot " + e);
var o = (p, t, e) => (v0(p, t, "read from private field"), e ? e.call(p) : t.get(p)), b = (p, t, e) => t.has(p) ? pE("Cannot add the same private member more than once") : t instanceof WeakSet ? t.add(p) : t.set(p, e), w = (p, t, e, n) => (v0(p, t, "write to private field"), n ? n.call(p, e) : t.set(p, e), e), E = (p, t, e) => (v0(p, t, "access private method"), e);
var Ge = (p, t, e, n) => ({
  set _(s) {
    w(p, t, s, e);
  },
  get _() {
    return o(p, t, n);
  }
});
function HT(p) {
  return p && p.__esModule && Object.prototype.hasOwnProperty.call(p, "default") ? p.default : p;
}
var w0 = { exports: {} }, vd = {}, A0 = { exports: {} }, zt = {};
/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var gE;
function jT() {
  if (gE) return zt;
  gE = 1;
  var p = Symbol.for("react.element"), t = Symbol.for("react.portal"), e = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), s = Symbol.for("react.profiler"), a = Symbol.for("react.provider"), l = Symbol.for("react.context"), u = Symbol.for("react.forward_ref"), h = Symbol.for("react.suspense"), f = Symbol.for("react.memo"), g = Symbol.for("react.lazy"), y = Symbol.iterator;
  function v(I) {
    return I === null || typeof I != "object" ? null : (I = y && I[y] || I["@@iterator"], typeof I == "function" ? I : null);
  }
  var S = { isMounted: function() {
    return !1;
  }, enqueueForceUpdate: function() {
  }, enqueueReplaceState: function() {
  }, enqueueSetState: function() {
  } }, x = Object.assign, _ = {};
  function k(I, z, K) {
    this.props = I, this.context = z, this.refs = _, this.updater = K || S;
  }
  k.prototype.isReactComponent = {}, k.prototype.setState = function(I, z) {
    if (typeof I != "object" && typeof I != "function" && I != null) throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
    this.updater.enqueueSetState(this, I, z, "setState");
  }, k.prototype.forceUpdate = function(I) {
    this.updater.enqueueForceUpdate(this, I, "forceUpdate");
  };
  function T() {
  }
  T.prototype = k.prototype;
  function R(I, z, K) {
    this.props = I, this.context = z, this.refs = _, this.updater = K || S;
  }
  var P = R.prototype = new T();
  P.constructor = R, x(P, k.prototype), P.isPureReactComponent = !0;
  var L = Array.isArray, D = Object.prototype.hasOwnProperty, N = { current: null }, B = { key: !0, ref: !0, __self: !0, __source: !0 };
  function $(I, z, K) {
    var ct, Et = {}, xt = null, bt = null;
    if (z != null) for (ct in z.ref !== void 0 && (bt = z.ref), z.key !== void 0 && (xt = "" + z.key), z) D.call(z, ct) && !B.hasOwnProperty(ct) && (Et[ct] = z[ct]);
    var Ot = arguments.length - 2;
    if (Ot === 1) Et.children = K;
    else if (1 < Ot) {
      for (var Bt = Array(Ot), Dt = 0; Dt < Ot; Dt++) Bt[Dt] = arguments[Dt + 2];
      Et.children = Bt;
    }
    if (I && I.defaultProps) for (ct in Ot = I.defaultProps, Ot) Et[ct] === void 0 && (Et[ct] = Ot[ct]);
    return { $$typeof: p, type: I, key: xt, ref: bt, props: Et, _owner: N.current };
  }
  function V(I, z) {
    return { $$typeof: p, type: I.type, key: z, ref: I.ref, props: I.props, _owner: I._owner };
  }
  function X(I) {
    return typeof I == "object" && I !== null && I.$$typeof === p;
  }
  function q(I) {
    var z = { "=": "=0", ":": "=2" };
    return "$" + I.replace(/[=:]/g, function(K) {
      return z[K];
    });
  }
  var it = /\/+/g;
  function lt(I, z) {
    return typeof I == "object" && I !== null && I.key != null ? q("" + I.key) : z.toString(36);
  }
  function rt(I, z, K, ct, Et) {
    var xt = typeof I;
    (xt === "undefined" || xt === "boolean") && (I = null);
    var bt = !1;
    if (I === null) bt = !0;
    else switch (xt) {
      case "string":
      case "number":
        bt = !0;
        break;
      case "object":
        switch (I.$$typeof) {
          case p:
          case t:
            bt = !0;
        }
    }
    if (bt) return bt = I, Et = Et(bt), I = ct === "" ? "." + lt(bt, 0) : ct, L(Et) ? (K = "", I != null && (K = I.replace(it, "$&/") + "/"), rt(Et, z, K, "", function(Dt) {
      return Dt;
    })) : Et != null && (X(Et) && (Et = V(Et, K + (!Et.key || bt && bt.key === Et.key ? "" : ("" + Et.key).replace(it, "$&/") + "/") + I)), z.push(Et)), 1;
    if (bt = 0, ct = ct === "" ? "." : ct + ":", L(I)) for (var Ot = 0; Ot < I.length; Ot++) {
      xt = I[Ot];
      var Bt = ct + lt(xt, Ot);
      bt += rt(xt, z, K, Bt, Et);
    }
    else if (Bt = v(I), typeof Bt == "function") for (I = Bt.call(I), Ot = 0; !(xt = I.next()).done; ) xt = xt.value, Bt = ct + lt(xt, Ot++), bt += rt(xt, z, K, Bt, Et);
    else if (xt === "object") throw z = String(I), Error("Objects are not valid as a React child (found: " + (z === "[object Object]" ? "object with keys {" + Object.keys(I).join(", ") + "}" : z) + "). If you meant to render a collection of children, use an array instead.");
    return bt;
  }
  function vt(I, z, K) {
    if (I == null) return I;
    var ct = [], Et = 0;
    return rt(I, ct, "", "", function(xt) {
      return z.call(K, xt, Et++);
    }), ct;
  }
  function tt(I) {
    if (I._status === -1) {
      var z = I._result;
      z = z(), z.then(function(K) {
        (I._status === 0 || I._status === -1) && (I._status = 1, I._result = K);
      }, function(K) {
        (I._status === 0 || I._status === -1) && (I._status = 2, I._result = K);
      }), I._status === -1 && (I._status = 0, I._result = z);
    }
    if (I._status === 1) return I._result.default;
    throw I._result;
  }
  var Q = { current: null }, W = { transition: null }, ot = { ReactCurrentDispatcher: Q, ReactCurrentBatchConfig: W, ReactCurrentOwner: N };
  function st() {
    throw Error("act(...) is not supported in production builds of React.");
  }
  return zt.Children = { map: vt, forEach: function(I, z, K) {
    vt(I, function() {
      z.apply(this, arguments);
    }, K);
  }, count: function(I) {
    var z = 0;
    return vt(I, function() {
      z++;
    }), z;
  }, toArray: function(I) {
    return vt(I, function(z) {
      return z;
    }) || [];
  }, only: function(I) {
    if (!X(I)) throw Error("React.Children.only expected to receive a single React element child.");
    return I;
  } }, zt.Component = k, zt.Fragment = e, zt.Profiler = s, zt.PureComponent = R, zt.StrictMode = n, zt.Suspense = h, zt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = ot, zt.act = st, zt.cloneElement = function(I, z, K) {
    if (I == null) throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + I + ".");
    var ct = x({}, I.props), Et = I.key, xt = I.ref, bt = I._owner;
    if (z != null) {
      if (z.ref !== void 0 && (xt = z.ref, bt = N.current), z.key !== void 0 && (Et = "" + z.key), I.type && I.type.defaultProps) var Ot = I.type.defaultProps;
      for (Bt in z) D.call(z, Bt) && !B.hasOwnProperty(Bt) && (ct[Bt] = z[Bt] === void 0 && Ot !== void 0 ? Ot[Bt] : z[Bt]);
    }
    var Bt = arguments.length - 2;
    if (Bt === 1) ct.children = K;
    else if (1 < Bt) {
      Ot = Array(Bt);
      for (var Dt = 0; Dt < Bt; Dt++) Ot[Dt] = arguments[Dt + 2];
      ct.children = Ot;
    }
    return { $$typeof: p, type: I.type, key: Et, ref: xt, props: ct, _owner: bt };
  }, zt.createContext = function(I) {
    return I = { $$typeof: l, _currentValue: I, _currentValue2: I, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null }, I.Provider = { $$typeof: a, _context: I }, I.Consumer = I;
  }, zt.createElement = $, zt.createFactory = function(I) {
    var z = $.bind(null, I);
    return z.type = I, z;
  }, zt.createRef = function() {
    return { current: null };
  }, zt.forwardRef = function(I) {
    return { $$typeof: u, render: I };
  }, zt.isValidElement = X, zt.lazy = function(I) {
    return { $$typeof: g, _payload: { _status: -1, _result: I }, _init: tt };
  }, zt.memo = function(I, z) {
    return { $$typeof: f, type: I, compare: z === void 0 ? null : z };
  }, zt.startTransition = function(I) {
    var z = W.transition;
    W.transition = {};
    try {
      I();
    } finally {
      W.transition = z;
    }
  }, zt.unstable_act = st, zt.useCallback = function(I, z) {
    return Q.current.useCallback(I, z);
  }, zt.useContext = function(I) {
    return Q.current.useContext(I);
  }, zt.useDebugValue = function() {
  }, zt.useDeferredValue = function(I) {
    return Q.current.useDeferredValue(I);
  }, zt.useEffect = function(I, z) {
    return Q.current.useEffect(I, z);
  }, zt.useId = function() {
    return Q.current.useId();
  }, zt.useImperativeHandle = function(I, z, K) {
    return Q.current.useImperativeHandle(I, z, K);
  }, zt.useInsertionEffect = function(I, z) {
    return Q.current.useInsertionEffect(I, z);
  }, zt.useLayoutEffect = function(I, z) {
    return Q.current.useLayoutEffect(I, z);
  }, zt.useMemo = function(I, z) {
    return Q.current.useMemo(I, z);
  }, zt.useReducer = function(I, z, K) {
    return Q.current.useReducer(I, z, K);
  }, zt.useRef = function(I) {
    return Q.current.useRef(I);
  }, zt.useState = function(I) {
    return Q.current.useState(I);
  }, zt.useSyncExternalStore = function(I, z, K) {
    return Q.current.useSyncExternalStore(I, z, K);
  }, zt.useTransition = function() {
    return Q.current.useTransition();
  }, zt.version = "18.3.1", zt;
}
var mE;
function aA() {
  return mE || (mE = 1, A0.exports = jT()), A0.exports;
}
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var yE;
function $T() {
  if (yE) return vd;
  yE = 1;
  var p = aA(), t = Symbol.for("react.element"), e = Symbol.for("react.fragment"), n = Object.prototype.hasOwnProperty, s = p.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, a = { key: !0, ref: !0, __self: !0, __source: !0 };
  function l(u, h, f) {
    var g, y = {}, v = null, S = null;
    f !== void 0 && (v = "" + f), h.key !== void 0 && (v = "" + h.key), h.ref !== void 0 && (S = h.ref);
    for (g in h) n.call(h, g) && !a.hasOwnProperty(g) && (y[g] = h[g]);
    if (u && u.defaultProps) for (g in h = u.defaultProps, h) y[g] === void 0 && (y[g] = h[g]);
    return { $$typeof: t, type: u, key: v, ref: S, props: y, _owner: s.current };
  }
  return vd.Fragment = e, vd.jsx = l, vd.jsxs = l, vd;
}
var vE;
function VT() {
  return vE || (vE = 1, w0.exports = $T()), w0.exports;
}
var St = VT(), nt = aA(), Jg = {}, S0 = { exports: {} }, Yn = {}, b0 = { exports: {} }, E0 = {};
/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var wE;
function WT() {
  return wE || (wE = 1, function(p) {
    function t(W, ot) {
      var st = W.length;
      W.push(ot);
      t: for (; 0 < st; ) {
        var I = st - 1 >>> 1, z = W[I];
        if (0 < s(z, ot)) W[I] = ot, W[st] = z, st = I;
        else break t;
      }
    }
    function e(W) {
      return W.length === 0 ? null : W[0];
    }
    function n(W) {
      if (W.length === 0) return null;
      var ot = W[0], st = W.pop();
      if (st !== ot) {
        W[0] = st;
        t: for (var I = 0, z = W.length, K = z >>> 1; I < K; ) {
          var ct = 2 * (I + 1) - 1, Et = W[ct], xt = ct + 1, bt = W[xt];
          if (0 > s(Et, st)) xt < z && 0 > s(bt, Et) ? (W[I] = bt, W[xt] = st, I = xt) : (W[I] = Et, W[ct] = st, I = ct);
          else if (xt < z && 0 > s(bt, st)) W[I] = bt, W[xt] = st, I = xt;
          else break t;
        }
      }
      return ot;
    }
    function s(W, ot) {
      var st = W.sortIndex - ot.sortIndex;
      return st !== 0 ? st : W.id - ot.id;
    }
    if (typeof performance == "object" && typeof performance.now == "function") {
      var a = performance;
      p.unstable_now = function() {
        return a.now();
      };
    } else {
      var l = Date, u = l.now();
      p.unstable_now = function() {
        return l.now() - u;
      };
    }
    var h = [], f = [], g = 1, y = null, v = 3, S = !1, x = !1, _ = !1, k = typeof setTimeout == "function" ? setTimeout : null, T = typeof clearTimeout == "function" ? clearTimeout : null, R = typeof setImmediate < "u" ? setImmediate : null;
    typeof navigator < "u" && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
    function P(W) {
      for (var ot = e(f); ot !== null; ) {
        if (ot.callback === null) n(f);
        else if (ot.startTime <= W) n(f), ot.sortIndex = ot.expirationTime, t(h, ot);
        else break;
        ot = e(f);
      }
    }
    function L(W) {
      if (_ = !1, P(W), !x) if (e(h) !== null) x = !0, tt(D);
      else {
        var ot = e(f);
        ot !== null && Q(L, ot.startTime - W);
      }
    }
    function D(W, ot) {
      x = !1, _ && (_ = !1, T($), $ = -1), S = !0;
      var st = v;
      try {
        for (P(ot), y = e(h); y !== null && (!(y.expirationTime > ot) || W && !q()); ) {
          var I = y.callback;
          if (typeof I == "function") {
            y.callback = null, v = y.priorityLevel;
            var z = I(y.expirationTime <= ot);
            ot = p.unstable_now(), typeof z == "function" ? y.callback = z : y === e(h) && n(h), P(ot);
          } else n(h);
          y = e(h);
        }
        if (y !== null) var K = !0;
        else {
          var ct = e(f);
          ct !== null && Q(L, ct.startTime - ot), K = !1;
        }
        return K;
      } finally {
        y = null, v = st, S = !1;
      }
    }
    var N = !1, B = null, $ = -1, V = 5, X = -1;
    function q() {
      return !(p.unstable_now() - X < V);
    }
    function it() {
      if (B !== null) {
        var W = p.unstable_now();
        X = W;
        var ot = !0;
        try {
          ot = B(!0, W);
        } finally {
          ot ? lt() : (N = !1, B = null);
        }
      } else N = !1;
    }
    var lt;
    if (typeof R == "function") lt = function() {
      R(it);
    };
    else if (typeof MessageChannel < "u") {
      var rt = new MessageChannel(), vt = rt.port2;
      rt.port1.onmessage = it, lt = function() {
        vt.postMessage(null);
      };
    } else lt = function() {
      k(it, 0);
    };
    function tt(W) {
      B = W, N || (N = !0, lt());
    }
    function Q(W, ot) {
      $ = k(function() {
        W(p.unstable_now());
      }, ot);
    }
    p.unstable_IdlePriority = 5, p.unstable_ImmediatePriority = 1, p.unstable_LowPriority = 4, p.unstable_NormalPriority = 3, p.unstable_Profiling = null, p.unstable_UserBlockingPriority = 2, p.unstable_cancelCallback = function(W) {
      W.callback = null;
    }, p.unstable_continueExecution = function() {
      x || S || (x = !0, tt(D));
    }, p.unstable_forceFrameRate = function(W) {
      0 > W || 125 < W ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : V = 0 < W ? Math.floor(1e3 / W) : 5;
    }, p.unstable_getCurrentPriorityLevel = function() {
      return v;
    }, p.unstable_getFirstCallbackNode = function() {
      return e(h);
    }, p.unstable_next = function(W) {
      switch (v) {
        case 1:
        case 2:
        case 3:
          var ot = 3;
          break;
        default:
          ot = v;
      }
      var st = v;
      v = ot;
      try {
        return W();
      } finally {
        v = st;
      }
    }, p.unstable_pauseExecution = function() {
    }, p.unstable_requestPaint = function() {
    }, p.unstable_runWithPriority = function(W, ot) {
      switch (W) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          break;
        default:
          W = 3;
      }
      var st = v;
      v = W;
      try {
        return ot();
      } finally {
        v = st;
      }
    }, p.unstable_scheduleCallback = function(W, ot, st) {
      var I = p.unstable_now();
      switch (typeof st == "object" && st !== null ? (st = st.delay, st = typeof st == "number" && 0 < st ? I + st : I) : st = I, W) {
        case 1:
          var z = -1;
          break;
        case 2:
          z = 250;
          break;
        case 5:
          z = 1073741823;
          break;
        case 4:
          z = 1e4;
          break;
        default:
          z = 5e3;
      }
      return z = st + z, W = { id: g++, callback: ot, priorityLevel: W, startTime: st, expirationTime: z, sortIndex: -1 }, st > I ? (W.sortIndex = st, t(f, W), e(h) === null && W === e(f) && (_ ? (T($), $ = -1) : _ = !0, Q(L, st - I))) : (W.sortIndex = z, t(h, W), x || S || (x = !0, tt(D))), W;
    }, p.unstable_shouldYield = q, p.unstable_wrapCallback = function(W) {
      var ot = v;
      return function() {
        var st = v;
        v = ot;
        try {
          return W.apply(this, arguments);
        } finally {
          v = st;
        }
      };
    };
  }(E0)), E0;
}
var AE;
function GT() {
  return AE || (AE = 1, b0.exports = WT()), b0.exports;
}
/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var SE;
function XT() {
  if (SE) return Yn;
  SE = 1;
  var p = aA(), t = GT();
  function e(i) {
    for (var r = "https://reactjs.org/docs/error-decoder.html?invariant=" + i, c = 1; c < arguments.length; c++) r += "&args[]=" + encodeURIComponent(arguments[c]);
    return "Minified React error #" + i + "; visit " + r + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  var n = /* @__PURE__ */ new Set(), s = {};
  function a(i, r) {
    l(i, r), l(i + "Capture", r);
  }
  function l(i, r) {
    for (s[i] = r, i = 0; i < r.length; i++) n.add(r[i]);
  }
  var u = !(typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u"), h = Object.prototype.hasOwnProperty, f = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, g = {}, y = {};
  function v(i) {
    return h.call(y, i) ? !0 : h.call(g, i) ? !1 : f.test(i) ? y[i] = !0 : (g[i] = !0, !1);
  }
  function S(i, r, c, d) {
    if (c !== null && c.type === 0) return !1;
    switch (typeof r) {
      case "function":
      case "symbol":
        return !0;
      case "boolean":
        return d ? !1 : c !== null ? !c.acceptsBooleans : (i = i.toLowerCase().slice(0, 5), i !== "data-" && i !== "aria-");
      default:
        return !1;
    }
  }
  function x(i, r, c, d) {
    if (r === null || typeof r > "u" || S(i, r, c, d)) return !0;
    if (d) return !1;
    if (c !== null) switch (c.type) {
      case 3:
        return !r;
      case 4:
        return r === !1;
      case 5:
        return isNaN(r);
      case 6:
        return isNaN(r) || 1 > r;
    }
    return !1;
  }
  function _(i, r, c, d, m, A, C) {
    this.acceptsBooleans = r === 2 || r === 3 || r === 4, this.attributeName = d, this.attributeNamespace = m, this.mustUseProperty = c, this.propertyName = i, this.type = r, this.sanitizeURL = A, this.removeEmptyString = C;
  }
  var k = {};
  "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(i) {
    k[i] = new _(i, 0, !1, i, null, !1, !1);
  }), [["acceptCharset", "accept-charset"], ["className", "class"], ["htmlFor", "for"], ["httpEquiv", "http-equiv"]].forEach(function(i) {
    var r = i[0];
    k[r] = new _(r, 1, !1, i[1], null, !1, !1);
  }), ["contentEditable", "draggable", "spellCheck", "value"].forEach(function(i) {
    k[i] = new _(i, 2, !1, i.toLowerCase(), null, !1, !1);
  }), ["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach(function(i) {
    k[i] = new _(i, 2, !1, i, null, !1, !1);
  }), "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(i) {
    k[i] = new _(i, 3, !1, i.toLowerCase(), null, !1, !1);
  }), ["checked", "multiple", "muted", "selected"].forEach(function(i) {
    k[i] = new _(i, 3, !0, i, null, !1, !1);
  }), ["capture", "download"].forEach(function(i) {
    k[i] = new _(i, 4, !1, i, null, !1, !1);
  }), ["cols", "rows", "size", "span"].forEach(function(i) {
    k[i] = new _(i, 6, !1, i, null, !1, !1);
  }), ["rowSpan", "start"].forEach(function(i) {
    k[i] = new _(i, 5, !1, i.toLowerCase(), null, !1, !1);
  });
  var T = /[\-:]([a-z])/g;
  function R(i) {
    return i[1].toUpperCase();
  }
  "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(i) {
    var r = i.replace(
      T,
      R
    );
    k[r] = new _(r, 1, !1, i, null, !1, !1);
  }), "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(i) {
    var r = i.replace(T, R);
    k[r] = new _(r, 1, !1, i, "http://www.w3.org/1999/xlink", !1, !1);
  }), ["xml:base", "xml:lang", "xml:space"].forEach(function(i) {
    var r = i.replace(T, R);
    k[r] = new _(r, 1, !1, i, "http://www.w3.org/XML/1998/namespace", !1, !1);
  }), ["tabIndex", "crossOrigin"].forEach(function(i) {
    k[i] = new _(i, 1, !1, i.toLowerCase(), null, !1, !1);
  }), k.xlinkHref = new _("xlinkHref", 1, !1, "xlink:href", "http://www.w3.org/1999/xlink", !0, !1), ["src", "href", "action", "formAction"].forEach(function(i) {
    k[i] = new _(i, 1, !1, i.toLowerCase(), null, !0, !0);
  });
  function P(i, r, c, d) {
    var m = k.hasOwnProperty(r) ? k[r] : null;
    (m !== null ? m.type !== 0 : d || !(2 < r.length) || r[0] !== "o" && r[0] !== "O" || r[1] !== "n" && r[1] !== "N") && (x(r, c, m, d) && (c = null), d || m === null ? v(r) && (c === null ? i.removeAttribute(r) : i.setAttribute(r, "" + c)) : m.mustUseProperty ? i[m.propertyName] = c === null ? m.type === 3 ? !1 : "" : c : (r = m.attributeName, d = m.attributeNamespace, c === null ? i.removeAttribute(r) : (m = m.type, c = m === 3 || m === 4 && c === !0 ? "" : "" + c, d ? i.setAttributeNS(d, r, c) : i.setAttribute(r, c))));
  }
  var L = p.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, D = Symbol.for("react.element"), N = Symbol.for("react.portal"), B = Symbol.for("react.fragment"), $ = Symbol.for("react.strict_mode"), V = Symbol.for("react.profiler"), X = Symbol.for("react.provider"), q = Symbol.for("react.context"), it = Symbol.for("react.forward_ref"), lt = Symbol.for("react.suspense"), rt = Symbol.for("react.suspense_list"), vt = Symbol.for("react.memo"), tt = Symbol.for("react.lazy"), Q = Symbol.for("react.offscreen"), W = Symbol.iterator;
  function ot(i) {
    return i === null || typeof i != "object" ? null : (i = W && i[W] || i["@@iterator"], typeof i == "function" ? i : null);
  }
  var st = Object.assign, I;
  function z(i) {
    if (I === void 0) try {
      throw Error();
    } catch (c) {
      var r = c.stack.trim().match(/\n( *(at )?)/);
      I = r && r[1] || "";
    }
    return `
` + I + i;
  }
  var K = !1;
  function ct(i, r) {
    if (!i || K) return "";
    K = !0;
    var c = Error.prepareStackTrace;
    Error.prepareStackTrace = void 0;
    try {
      if (r) if (r = function() {
        throw Error();
      }, Object.defineProperty(r.prototype, "props", { set: function() {
        throw Error();
      } }), typeof Reflect == "object" && Reflect.construct) {
        try {
          Reflect.construct(r, []);
        } catch (j) {
          var d = j;
        }
        Reflect.construct(i, [], r);
      } else {
        try {
          r.call();
        } catch (j) {
          d = j;
        }
        i.call(r.prototype);
      }
      else {
        try {
          throw Error();
        } catch (j) {
          d = j;
        }
        i();
      }
    } catch (j) {
      if (j && d && typeof j.stack == "string") {
        for (var m = j.stack.split(`
`), A = d.stack.split(`
`), C = m.length - 1, M = A.length - 1; 1 <= C && 0 <= M && m[C] !== A[M]; ) M--;
        for (; 1 <= C && 0 <= M; C--, M--) if (m[C] !== A[M]) {
          if (C !== 1 || M !== 1)
            do
              if (C--, M--, 0 > M || m[C] !== A[M]) {
                var F = `
` + m[C].replace(" at new ", " at ");
                return i.displayName && F.includes("<anonymous>") && (F = F.replace("<anonymous>", i.displayName)), F;
              }
            while (1 <= C && 0 <= M);
          break;
        }
      }
    } finally {
      K = !1, Error.prepareStackTrace = c;
    }
    return (i = i ? i.displayName || i.name : "") ? z(i) : "";
  }
  function Et(i) {
    switch (i.tag) {
      case 5:
        return z(i.type);
      case 16:
        return z("Lazy");
      case 13:
        return z("Suspense");
      case 19:
        return z("SuspenseList");
      case 0:
      case 2:
      case 15:
        return i = ct(i.type, !1), i;
      case 11:
        return i = ct(i.type.render, !1), i;
      case 1:
        return i = ct(i.type, !0), i;
      default:
        return "";
    }
  }
  function xt(i) {
    if (i == null) return null;
    if (typeof i == "function") return i.displayName || i.name || null;
    if (typeof i == "string") return i;
    switch (i) {
      case B:
        return "Fragment";
      case N:
        return "Portal";
      case V:
        return "Profiler";
      case $:
        return "StrictMode";
      case lt:
        return "Suspense";
      case rt:
        return "SuspenseList";
    }
    if (typeof i == "object") switch (i.$$typeof) {
      case q:
        return (i.displayName || "Context") + ".Consumer";
      case X:
        return (i._context.displayName || "Context") + ".Provider";
      case it:
        var r = i.render;
        return i = i.displayName, i || (i = r.displayName || r.name || "", i = i !== "" ? "ForwardRef(" + i + ")" : "ForwardRef"), i;
      case vt:
        return r = i.displayName || null, r !== null ? r : xt(i.type) || "Memo";
      case tt:
        r = i._payload, i = i._init;
        try {
          return xt(i(r));
        } catch {
        }
    }
    return null;
  }
  function bt(i) {
    var r = i.type;
    switch (i.tag) {
      case 24:
        return "Cache";
      case 9:
        return (r.displayName || "Context") + ".Consumer";
      case 10:
        return (r._context.displayName || "Context") + ".Provider";
      case 18:
        return "DehydratedFragment";
      case 11:
        return i = r.render, i = i.displayName || i.name || "", r.displayName || (i !== "" ? "ForwardRef(" + i + ")" : "ForwardRef");
      case 7:
        return "Fragment";
      case 5:
        return r;
      case 4:
        return "Portal";
      case 3:
        return "Root";
      case 6:
        return "Text";
      case 16:
        return xt(r);
      case 8:
        return r === $ ? "StrictMode" : "Mode";
      case 22:
        return "Offscreen";
      case 12:
        return "Profiler";
      case 21:
        return "Scope";
      case 13:
        return "Suspense";
      case 19:
        return "SuspenseList";
      case 25:
        return "TracingMarker";
      case 1:
      case 0:
      case 17:
      case 2:
      case 14:
      case 15:
        if (typeof r == "function") return r.displayName || r.name || null;
        if (typeof r == "string") return r;
    }
    return null;
  }
  function Ot(i) {
    switch (typeof i) {
      case "boolean":
      case "number":
      case "string":
      case "undefined":
        return i;
      case "object":
        return i;
      default:
        return "";
    }
  }
  function Bt(i) {
    var r = i.type;
    return (i = i.nodeName) && i.toLowerCase() === "input" && (r === "checkbox" || r === "radio");
  }
  function Dt(i) {
    var r = Bt(i) ? "checked" : "value", c = Object.getOwnPropertyDescriptor(i.constructor.prototype, r), d = "" + i[r];
    if (!i.hasOwnProperty(r) && typeof c < "u" && typeof c.get == "function" && typeof c.set == "function") {
      var m = c.get, A = c.set;
      return Object.defineProperty(i, r, { configurable: !0, get: function() {
        return m.call(this);
      }, set: function(C) {
        d = "" + C, A.call(this, C);
      } }), Object.defineProperty(i, r, { enumerable: c.enumerable }), { getValue: function() {
        return d;
      }, setValue: function(C) {
        d = "" + C;
      }, stopTracking: function() {
        i._valueTracker = null, delete i[r];
      } };
    }
  }
  function Qe(i) {
    i._valueTracker || (i._valueTracker = Dt(i));
  }
  function Di(i) {
    if (!i) return !1;
    var r = i._valueTracker;
    if (!r) return !0;
    var c = r.getValue(), d = "";
    return i && (d = Bt(i) ? i.checked ? "true" : "false" : i.value), i = d, i !== c ? (r.setValue(i), !0) : !1;
  }
  function Ht(i) {
    if (i = i || (typeof document < "u" ? document : void 0), typeof i > "u") return null;
    try {
      return i.activeElement || i.body;
    } catch {
      return i.body;
    }
  }
  function Jt(i, r) {
    var c = r.checked;
    return st({}, r, { defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: c ?? i._wrapperState.initialChecked });
  }
  function Te(i, r) {
    var c = r.defaultValue == null ? "" : r.defaultValue, d = r.checked != null ? r.checked : r.defaultChecked;
    c = Ot(r.value != null ? r.value : c), i._wrapperState = { initialChecked: d, initialValue: c, controlled: r.type === "checkbox" || r.type === "radio" ? r.checked != null : r.value != null };
  }
  function Ie(i, r) {
    r = r.checked, r != null && P(i, "checked", r, !1);
  }
  function Rs(i, r) {
    Ie(i, r);
    var c = Ot(r.value), d = r.type;
    if (c != null) d === "number" ? (c === 0 && i.value === "" || i.value != c) && (i.value = "" + c) : i.value !== "" + c && (i.value = "" + c);
    else if (d === "submit" || d === "reset") {
      i.removeAttribute("value");
      return;
    }
    r.hasOwnProperty("value") ? Ma(i, r.type, c) : r.hasOwnProperty("defaultValue") && Ma(i, r.type, Ot(r.defaultValue)), r.checked == null && r.defaultChecked != null && (i.defaultChecked = !!r.defaultChecked);
  }
  function ho(i, r, c) {
    if (r.hasOwnProperty("value") || r.hasOwnProperty("defaultValue")) {
      var d = r.type;
      if (!(d !== "submit" && d !== "reset" || r.value !== void 0 && r.value !== null)) return;
      r = "" + i._wrapperState.initialValue, c || r === i.value || (i.value = r), i.defaultValue = r;
    }
    c = i.name, c !== "" && (i.name = ""), i.defaultChecked = !!i._wrapperState.initialChecked, c !== "" && (i.name = c);
  }
  function Ma(i, r, c) {
    (r !== "number" || Ht(i.ownerDocument) !== i) && (c == null ? i.defaultValue = "" + i._wrapperState.initialValue : i.defaultValue !== "" + c && (i.defaultValue = "" + c));
  }
  var La = Array.isArray;
  function is(i, r, c, d) {
    if (i = i.options, r) {
      r = {};
      for (var m = 0; m < c.length; m++) r["$" + c[m]] = !0;
      for (c = 0; c < i.length; c++) m = r.hasOwnProperty("$" + i[c].value), i[c].selected !== m && (i[c].selected = m), m && d && (i[c].defaultSelected = !0);
    } else {
      for (c = "" + Ot(c), r = null, m = 0; m < i.length; m++) {
        if (i[m].value === c) {
          i[m].selected = !0, d && (i[m].defaultSelected = !0);
          return;
        }
        r !== null || i[m].disabled || (r = i[m]);
      }
      r !== null && (r.selected = !0);
    }
  }
  function Ih(i, r) {
    if (r.dangerouslySetInnerHTML != null) throw Error(e(91));
    return st({}, r, { value: void 0, defaultValue: void 0, children: "" + i._wrapperState.initialValue });
  }
  function jp(i, r) {
    var c = r.value;
    if (c == null) {
      if (c = r.children, r = r.defaultValue, c != null) {
        if (r != null) throw Error(e(92));
        if (La(c)) {
          if (1 < c.length) throw Error(e(93));
          c = c[0];
        }
        r = c;
      }
      r == null && (r = ""), c = r;
    }
    i._wrapperState = { initialValue: Ot(c) };
  }
  function $p(i, r) {
    var c = Ot(r.value), d = Ot(r.defaultValue);
    c != null && (c = "" + c, c !== i.value && (i.value = c), r.defaultValue == null && i.defaultValue !== c && (i.defaultValue = c)), d != null && (i.defaultValue = "" + d);
  }
  function Vp(i) {
    var r = i.textContent;
    r === i._wrapperState.initialValue && r !== "" && r !== null && (i.value = r);
  }
  function Wp(i) {
    switch (i) {
      case "svg":
        return "http://www.w3.org/2000/svg";
      case "math":
        return "http://www.w3.org/1998/Math/MathML";
      default:
        return "http://www.w3.org/1999/xhtml";
    }
  }
  function Ii(i, r) {
    return i == null || i === "http://www.w3.org/1999/xhtml" ? Wp(r) : i === "http://www.w3.org/2000/svg" && r === "foreignObject" ? "http://www.w3.org/1999/xhtml" : i;
  }
  var ss, rs = function(i) {
    return typeof MSApp < "u" && MSApp.execUnsafeLocalFunction ? function(r, c, d, m) {
      MSApp.execUnsafeLocalFunction(function() {
        return i(r, c, d, m);
      });
    } : i;
  }(function(i, r) {
    if (i.namespaceURI !== "http://www.w3.org/2000/svg" || "innerHTML" in i) i.innerHTML = r;
    else {
      for (ss = ss || document.createElement("div"), ss.innerHTML = "<svg>" + r.valueOf().toString() + "</svg>", r = ss.firstChild; i.firstChild; ) i.removeChild(i.firstChild);
      for (; r.firstChild; ) i.appendChild(r.firstChild);
    }
  });
  function hr(i, r) {
    if (r) {
      var c = i.firstChild;
      if (c && c === i.lastChild && c.nodeType === 3) {
        c.nodeValue = r;
        return;
      }
    }
    i.textContent = r;
  }
  var Fh = {
    animationIterationCount: !0,
    aspectRatio: !0,
    borderImageOutset: !0,
    borderImageSlice: !0,
    borderImageWidth: !0,
    boxFlex: !0,
    boxFlexGroup: !0,
    boxOrdinalGroup: !0,
    columnCount: !0,
    columns: !0,
    flex: !0,
    flexGrow: !0,
    flexPositive: !0,
    flexShrink: !0,
    flexNegative: !0,
    flexOrder: !0,
    gridArea: !0,
    gridRow: !0,
    gridRowEnd: !0,
    gridRowSpan: !0,
    gridRowStart: !0,
    gridColumn: !0,
    gridColumnEnd: !0,
    gridColumnSpan: !0,
    gridColumnStart: !0,
    fontWeight: !0,
    lineClamp: !0,
    lineHeight: !0,
    opacity: !0,
    order: !0,
    orphans: !0,
    tabSize: !0,
    widows: !0,
    zIndex: !0,
    zoom: !0,
    fillOpacity: !0,
    floodOpacity: !0,
    stopOpacity: !0,
    strokeDasharray: !0,
    strokeDashoffset: !0,
    strokeMiterlimit: !0,
    strokeOpacity: !0,
    strokeWidth: !0
  }, $C = ["Webkit", "ms", "Moz", "O"];
  Object.keys(Fh).forEach(function(i) {
    $C.forEach(function(r) {
      r = r + i.charAt(0).toUpperCase() + i.substring(1), Fh[r] = Fh[i];
    });
  });
  function PA(i, r, c) {
    return r == null || typeof r == "boolean" || r === "" ? "" : c || typeof r != "number" || r === 0 || Fh.hasOwnProperty(i) && Fh[i] ? ("" + r).trim() : r + "px";
  }
  function RA(i, r) {
    i = i.style;
    for (var c in r) if (r.hasOwnProperty(c)) {
      var d = c.indexOf("--") === 0, m = PA(c, r[c], d);
      c === "float" && (c = "cssFloat"), d ? i.setProperty(c, m) : i[c] = m;
    }
  }
  var VC = st({ menuitem: !0 }, { area: !0, base: !0, br: !0, col: !0, embed: !0, hr: !0, img: !0, input: !0, keygen: !0, link: !0, meta: !0, param: !0, source: !0, track: !0, wbr: !0 });
  function Ry(i, r) {
    if (r) {
      if (VC[i] && (r.children != null || r.dangerouslySetInnerHTML != null)) throw Error(e(137, i));
      if (r.dangerouslySetInnerHTML != null) {
        if (r.children != null) throw Error(e(60));
        if (typeof r.dangerouslySetInnerHTML != "object" || !("__html" in r.dangerouslySetInnerHTML)) throw Error(e(61));
      }
      if (r.style != null && typeof r.style != "object") throw Error(e(62));
    }
  }
  function My(i, r) {
    if (i.indexOf("-") === -1) return typeof r.is == "string";
    switch (i) {
      case "annotation-xml":
      case "color-profile":
      case "font-face":
      case "font-face-src":
      case "font-face-uri":
      case "font-face-format":
      case "font-face-name":
      case "missing-glyph":
        return !1;
      default:
        return !0;
    }
  }
  var Ly = null;
  function Dy(i) {
    return i = i.target || i.srcElement || window, i.correspondingUseElement && (i = i.correspondingUseElement), i.nodeType === 3 ? i.parentNode : i;
  }
  var Iy = null, gc = null, mc = null;
  function MA(i) {
    if (i = id(i)) {
      if (typeof Iy != "function") throw Error(e(280));
      var r = i.stateNode;
      r && (r = pg(r), Iy(i.stateNode, i.type, r));
    }
  }
  function LA(i) {
    gc ? mc ? mc.push(i) : mc = [i] : gc = i;
  }
  function DA() {
    if (gc) {
      var i = gc, r = mc;
      if (mc = gc = null, MA(i), r) for (i = 0; i < r.length; i++) MA(r[i]);
    }
  }
  function IA(i, r) {
    return i(r);
  }
  function FA() {
  }
  var Fy = !1;
  function NA(i, r, c) {
    if (Fy) return i(r, c);
    Fy = !0;
    try {
      return IA(i, r, c);
    } finally {
      Fy = !1, (gc !== null || mc !== null) && (FA(), DA());
    }
  }
  function Nh(i, r) {
    var c = i.stateNode;
    if (c === null) return null;
    var d = pg(c);
    if (d === null) return null;
    c = d[r];
    t: switch (r) {
      case "onClick":
      case "onClickCapture":
      case "onDoubleClick":
      case "onDoubleClickCapture":
      case "onMouseDown":
      case "onMouseDownCapture":
      case "onMouseMove":
      case "onMouseMoveCapture":
      case "onMouseUp":
      case "onMouseUpCapture":
      case "onMouseEnter":
        (d = !d.disabled) || (i = i.type, d = !(i === "button" || i === "input" || i === "select" || i === "textarea")), i = !d;
        break t;
      default:
        i = !1;
    }
    if (i) return null;
    if (c && typeof c != "function") throw Error(e(231, r, typeof c));
    return c;
  }
  var Ny = !1;
  if (u) try {
    var Oh = {};
    Object.defineProperty(Oh, "passive", { get: function() {
      Ny = !0;
    } }), window.addEventListener("test", Oh, Oh), window.removeEventListener("test", Oh, Oh);
  } catch {
    Ny = !1;
  }
  function WC(i, r, c, d, m, A, C, M, F) {
    var j = Array.prototype.slice.call(arguments, 3);
    try {
      r.apply(c, j);
    } catch (Y) {
      this.onError(Y);
    }
  }
  var Bh = !1, Gp = null, Xp = !1, Oy = null, GC = { onError: function(i) {
    Bh = !0, Gp = i;
  } };
  function XC(i, r, c, d, m, A, C, M, F) {
    Bh = !1, Gp = null, WC.apply(GC, arguments);
  }
  function qC(i, r, c, d, m, A, C, M, F) {
    if (XC.apply(this, arguments), Bh) {
      if (Bh) {
        var j = Gp;
        Bh = !1, Gp = null;
      } else throw Error(e(198));
      Xp || (Xp = !0, Oy = j);
    }
  }
  function Da(i) {
    var r = i, c = i;
    if (i.alternate) for (; r.return; ) r = r.return;
    else {
      i = r;
      do
        r = i, (r.flags & 4098) !== 0 && (c = r.return), i = r.return;
      while (i);
    }
    return r.tag === 3 ? c : null;
  }
  function OA(i) {
    if (i.tag === 13) {
      var r = i.memoizedState;
      if (r === null && (i = i.alternate, i !== null && (r = i.memoizedState)), r !== null) return r.dehydrated;
    }
    return null;
  }
  function BA(i) {
    if (Da(i) !== i) throw Error(e(188));
  }
  function YC(i) {
    var r = i.alternate;
    if (!r) {
      if (r = Da(i), r === null) throw Error(e(188));
      return r !== i ? null : i;
    }
    for (var c = i, d = r; ; ) {
      var m = c.return;
      if (m === null) break;
      var A = m.alternate;
      if (A === null) {
        if (d = m.return, d !== null) {
          c = d;
          continue;
        }
        break;
      }
      if (m.child === A.child) {
        for (A = m.child; A; ) {
          if (A === c) return BA(m), i;
          if (A === d) return BA(m), r;
          A = A.sibling;
        }
        throw Error(e(188));
      }
      if (c.return !== d.return) c = m, d = A;
      else {
        for (var C = !1, M = m.child; M; ) {
          if (M === c) {
            C = !0, c = m, d = A;
            break;
          }
          if (M === d) {
            C = !0, d = m, c = A;
            break;
          }
          M = M.sibling;
        }
        if (!C) {
          for (M = A.child; M; ) {
            if (M === c) {
              C = !0, c = A, d = m;
              break;
            }
            if (M === d) {
              C = !0, d = A, c = m;
              break;
            }
            M = M.sibling;
          }
          if (!C) throw Error(e(189));
        }
      }
      if (c.alternate !== d) throw Error(e(190));
    }
    if (c.tag !== 3) throw Error(e(188));
    return c.stateNode.current === c ? i : r;
  }
  function zA(i) {
    return i = YC(i), i !== null ? UA(i) : null;
  }
  function UA(i) {
    if (i.tag === 5 || i.tag === 6) return i;
    for (i = i.child; i !== null; ) {
      var r = UA(i);
      if (r !== null) return r;
      i = i.sibling;
    }
    return null;
  }
  var HA = t.unstable_scheduleCallback, jA = t.unstable_cancelCallback, KC = t.unstable_shouldYield, QC = t.unstable_requestPaint, Fe = t.unstable_now, ZC = t.unstable_getCurrentPriorityLevel, By = t.unstable_ImmediatePriority, $A = t.unstable_UserBlockingPriority, qp = t.unstable_NormalPriority, JC = t.unstable_LowPriority, VA = t.unstable_IdlePriority, Yp = null, Ms = null;
  function tx(i) {
    if (Ms && typeof Ms.onCommitFiberRoot == "function") try {
      Ms.onCommitFiberRoot(Yp, i, void 0, (i.current.flags & 128) === 128);
    } catch {
    }
  }
  var os = Math.clz32 ? Math.clz32 : ix, ex = Math.log, nx = Math.LN2;
  function ix(i) {
    return i >>>= 0, i === 0 ? 32 : 31 - (ex(i) / nx | 0) | 0;
  }
  var Kp = 64, Qp = 4194304;
  function zh(i) {
    switch (i & -i) {
      case 1:
        return 1;
      case 2:
        return 2;
      case 4:
        return 4;
      case 8:
        return 8;
      case 16:
        return 16;
      case 32:
        return 32;
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return i & 4194240;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return i & 130023424;
      case 134217728:
        return 134217728;
      case 268435456:
        return 268435456;
      case 536870912:
        return 536870912;
      case 1073741824:
        return 1073741824;
      default:
        return i;
    }
  }
  function Zp(i, r) {
    var c = i.pendingLanes;
    if (c === 0) return 0;
    var d = 0, m = i.suspendedLanes, A = i.pingedLanes, C = c & 268435455;
    if (C !== 0) {
      var M = C & ~m;
      M !== 0 ? d = zh(M) : (A &= C, A !== 0 && (d = zh(A)));
    } else C = c & ~m, C !== 0 ? d = zh(C) : A !== 0 && (d = zh(A));
    if (d === 0) return 0;
    if (r !== 0 && r !== d && (r & m) === 0 && (m = d & -d, A = r & -r, m >= A || m === 16 && (A & 4194240) !== 0)) return r;
    if ((d & 4) !== 0 && (d |= c & 16), r = i.entangledLanes, r !== 0) for (i = i.entanglements, r &= d; 0 < r; ) c = 31 - os(r), m = 1 << c, d |= i[c], r &= ~m;
    return d;
  }
  function sx(i, r) {
    switch (i) {
      case 1:
      case 2:
      case 4:
        return r + 250;
      case 8:
      case 16:
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
      case 1024:
      case 2048:
      case 4096:
      case 8192:
      case 16384:
      case 32768:
      case 65536:
      case 131072:
      case 262144:
      case 524288:
      case 1048576:
      case 2097152:
        return r + 5e3;
      case 4194304:
      case 8388608:
      case 16777216:
      case 33554432:
      case 67108864:
        return -1;
      case 134217728:
      case 268435456:
      case 536870912:
      case 1073741824:
        return -1;
      default:
        return -1;
    }
  }
  function rx(i, r) {
    for (var c = i.suspendedLanes, d = i.pingedLanes, m = i.expirationTimes, A = i.pendingLanes; 0 < A; ) {
      var C = 31 - os(A), M = 1 << C, F = m[C];
      F === -1 ? ((M & c) === 0 || (M & d) !== 0) && (m[C] = sx(M, r)) : F <= r && (i.expiredLanes |= M), A &= ~M;
    }
  }
  function zy(i) {
    return i = i.pendingLanes & -1073741825, i !== 0 ? i : i & 1073741824 ? 1073741824 : 0;
  }
  function WA() {
    var i = Kp;
    return Kp <<= 1, (Kp & 4194240) === 0 && (Kp = 64), i;
  }
  function Uy(i) {
    for (var r = [], c = 0; 31 > c; c++) r.push(i);
    return r;
  }
  function Uh(i, r, c) {
    i.pendingLanes |= r, r !== 536870912 && (i.suspendedLanes = 0, i.pingedLanes = 0), i = i.eventTimes, r = 31 - os(r), i[r] = c;
  }
  function ox(i, r) {
    var c = i.pendingLanes & ~r;
    i.pendingLanes = r, i.suspendedLanes = 0, i.pingedLanes = 0, i.expiredLanes &= r, i.mutableReadLanes &= r, i.entangledLanes &= r, r = i.entanglements;
    var d = i.eventTimes;
    for (i = i.expirationTimes; 0 < c; ) {
      var m = 31 - os(c), A = 1 << m;
      r[m] = 0, d[m] = -1, i[m] = -1, c &= ~A;
    }
  }
  function Hy(i, r) {
    var c = i.entangledLanes |= r;
    for (i = i.entanglements; c; ) {
      var d = 31 - os(c), m = 1 << d;
      m & r | i[d] & r && (i[d] |= r), c &= ~m;
    }
  }
  var Kt = 0;
  function GA(i) {
    return i &= -i, 1 < i ? 4 < i ? (i & 268435455) !== 0 ? 16 : 536870912 : 4 : 1;
  }
  var XA, jy, qA, YA, KA, $y = !1, Jp = [], fo = null, po = null, go = null, Hh = /* @__PURE__ */ new Map(), jh = /* @__PURE__ */ new Map(), mo = [], ax = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
  function QA(i, r) {
    switch (i) {
      case "focusin":
      case "focusout":
        fo = null;
        break;
      case "dragenter":
      case "dragleave":
        po = null;
        break;
      case "mouseover":
      case "mouseout":
        go = null;
        break;
      case "pointerover":
      case "pointerout":
        Hh.delete(r.pointerId);
        break;
      case "gotpointercapture":
      case "lostpointercapture":
        jh.delete(r.pointerId);
    }
  }
  function $h(i, r, c, d, m, A) {
    return i === null || i.nativeEvent !== A ? (i = { blockedOn: r, domEventName: c, eventSystemFlags: d, nativeEvent: A, targetContainers: [m] }, r !== null && (r = id(r), r !== null && jy(r)), i) : (i.eventSystemFlags |= d, r = i.targetContainers, m !== null && r.indexOf(m) === -1 && r.push(m), i);
  }
  function lx(i, r, c, d, m) {
    switch (r) {
      case "focusin":
        return fo = $h(fo, i, r, c, d, m), !0;
      case "dragenter":
        return po = $h(po, i, r, c, d, m), !0;
      case "mouseover":
        return go = $h(go, i, r, c, d, m), !0;
      case "pointerover":
        var A = m.pointerId;
        return Hh.set(A, $h(Hh.get(A) || null, i, r, c, d, m)), !0;
      case "gotpointercapture":
        return A = m.pointerId, jh.set(A, $h(jh.get(A) || null, i, r, c, d, m)), !0;
    }
    return !1;
  }
  function ZA(i) {
    var r = Ia(i.target);
    if (r !== null) {
      var c = Da(r);
      if (c !== null) {
        if (r = c.tag, r === 13) {
          if (r = OA(c), r !== null) {
            i.blockedOn = r, KA(i.priority, function() {
              qA(c);
            });
            return;
          }
        } else if (r === 3 && c.stateNode.current.memoizedState.isDehydrated) {
          i.blockedOn = c.tag === 3 ? c.stateNode.containerInfo : null;
          return;
        }
      }
    }
    i.blockedOn = null;
  }
  function tg(i) {
    if (i.blockedOn !== null) return !1;
    for (var r = i.targetContainers; 0 < r.length; ) {
      var c = Wy(i.domEventName, i.eventSystemFlags, r[0], i.nativeEvent);
      if (c === null) {
        c = i.nativeEvent;
        var d = new c.constructor(c.type, c);
        Ly = d, c.target.dispatchEvent(d), Ly = null;
      } else return r = id(c), r !== null && jy(r), i.blockedOn = c, !1;
      r.shift();
    }
    return !0;
  }
  function JA(i, r, c) {
    tg(i) && c.delete(r);
  }
  function cx() {
    $y = !1, fo !== null && tg(fo) && (fo = null), po !== null && tg(po) && (po = null), go !== null && tg(go) && (go = null), Hh.forEach(JA), jh.forEach(JA);
  }
  function Vh(i, r) {
    i.blockedOn === r && (i.blockedOn = null, $y || ($y = !0, t.unstable_scheduleCallback(t.unstable_NormalPriority, cx)));
  }
  function Wh(i) {
    function r(m) {
      return Vh(m, i);
    }
    if (0 < Jp.length) {
      Vh(Jp[0], i);
      for (var c = 1; c < Jp.length; c++) {
        var d = Jp[c];
        d.blockedOn === i && (d.blockedOn = null);
      }
    }
    for (fo !== null && Vh(fo, i), po !== null && Vh(po, i), go !== null && Vh(go, i), Hh.forEach(r), jh.forEach(r), c = 0; c < mo.length; c++) d = mo[c], d.blockedOn === i && (d.blockedOn = null);
    for (; 0 < mo.length && (c = mo[0], c.blockedOn === null); ) ZA(c), c.blockedOn === null && mo.shift();
  }
  var yc = L.ReactCurrentBatchConfig, eg = !0;
  function ux(i, r, c, d) {
    var m = Kt, A = yc.transition;
    yc.transition = null;
    try {
      Kt = 1, Vy(i, r, c, d);
    } finally {
      Kt = m, yc.transition = A;
    }
  }
  function hx(i, r, c, d) {
    var m = Kt, A = yc.transition;
    yc.transition = null;
    try {
      Kt = 4, Vy(i, r, c, d);
    } finally {
      Kt = m, yc.transition = A;
    }
  }
  function Vy(i, r, c, d) {
    if (eg) {
      var m = Wy(i, r, c, d);
      if (m === null) lv(i, r, d, ng, c), QA(i, d);
      else if (lx(m, i, r, c, d)) d.stopPropagation();
      else if (QA(i, d), r & 4 && -1 < ax.indexOf(i)) {
        for (; m !== null; ) {
          var A = id(m);
          if (A !== null && XA(A), A = Wy(i, r, c, d), A === null && lv(i, r, d, ng, c), A === m) break;
          m = A;
        }
        m !== null && d.stopPropagation();
      } else lv(i, r, d, null, c);
    }
  }
  var ng = null;
  function Wy(i, r, c, d) {
    if (ng = null, i = Dy(d), i = Ia(i), i !== null) if (r = Da(i), r === null) i = null;
    else if (c = r.tag, c === 13) {
      if (i = OA(r), i !== null) return i;
      i = null;
    } else if (c === 3) {
      if (r.stateNode.current.memoizedState.isDehydrated) return r.tag === 3 ? r.stateNode.containerInfo : null;
      i = null;
    } else r !== i && (i = null);
    return ng = i, null;
  }
  function tS(i) {
    switch (i) {
      case "cancel":
      case "click":
      case "close":
      case "contextmenu":
      case "copy":
      case "cut":
      case "auxclick":
      case "dblclick":
      case "dragend":
      case "dragstart":
      case "drop":
      case "focusin":
      case "focusout":
      case "input":
      case "invalid":
      case "keydown":
      case "keypress":
      case "keyup":
      case "mousedown":
      case "mouseup":
      case "paste":
      case "pause":
      case "play":
      case "pointercancel":
      case "pointerdown":
      case "pointerup":
      case "ratechange":
      case "reset":
      case "resize":
      case "seeked":
      case "submit":
      case "touchcancel":
      case "touchend":
      case "touchstart":
      case "volumechange":
      case "change":
      case "selectionchange":
      case "textInput":
      case "compositionstart":
      case "compositionend":
      case "compositionupdate":
      case "beforeblur":
      case "afterblur":
      case "beforeinput":
      case "blur":
      case "fullscreenchange":
      case "focus":
      case "hashchange":
      case "popstate":
      case "select":
      case "selectstart":
        return 1;
      case "drag":
      case "dragenter":
      case "dragexit":
      case "dragleave":
      case "dragover":
      case "mousemove":
      case "mouseout":
      case "mouseover":
      case "pointermove":
      case "pointerout":
      case "pointerover":
      case "scroll":
      case "toggle":
      case "touchmove":
      case "wheel":
      case "mouseenter":
      case "mouseleave":
      case "pointerenter":
      case "pointerleave":
        return 4;
      case "message":
        switch (ZC()) {
          case By:
            return 1;
          case $A:
            return 4;
          case qp:
          case JC:
            return 16;
          case VA:
            return 536870912;
          default:
            return 16;
        }
      default:
        return 16;
    }
  }
  var yo = null, Gy = null, ig = null;
  function eS() {
    if (ig) return ig;
    var i, r = Gy, c = r.length, d, m = "value" in yo ? yo.value : yo.textContent, A = m.length;
    for (i = 0; i < c && r[i] === m[i]; i++) ;
    var C = c - i;
    for (d = 1; d <= C && r[c - d] === m[A - d]; d++) ;
    return ig = m.slice(i, 1 < d ? 1 - d : void 0);
  }
  function sg(i) {
    var r = i.keyCode;
    return "charCode" in i ? (i = i.charCode, i === 0 && r === 13 && (i = 13)) : i = r, i === 10 && (i = 13), 32 <= i || i === 13 ? i : 0;
  }
  function rg() {
    return !0;
  }
  function nS() {
    return !1;
  }
  function oi(i) {
    function r(c, d, m, A, C) {
      this._reactName = c, this._targetInst = m, this.type = d, this.nativeEvent = A, this.target = C, this.currentTarget = null;
      for (var M in i) i.hasOwnProperty(M) && (c = i[M], this[M] = c ? c(A) : A[M]);
      return this.isDefaultPrevented = (A.defaultPrevented != null ? A.defaultPrevented : A.returnValue === !1) ? rg : nS, this.isPropagationStopped = nS, this;
    }
    return st(r.prototype, { preventDefault: function() {
      this.defaultPrevented = !0;
      var c = this.nativeEvent;
      c && (c.preventDefault ? c.preventDefault() : typeof c.returnValue != "unknown" && (c.returnValue = !1), this.isDefaultPrevented = rg);
    }, stopPropagation: function() {
      var c = this.nativeEvent;
      c && (c.stopPropagation ? c.stopPropagation() : typeof c.cancelBubble != "unknown" && (c.cancelBubble = !0), this.isPropagationStopped = rg);
    }, persist: function() {
    }, isPersistent: rg }), r;
  }
  var vc = { eventPhase: 0, bubbles: 0, cancelable: 0, timeStamp: function(i) {
    return i.timeStamp || Date.now();
  }, defaultPrevented: 0, isTrusted: 0 }, Xy = oi(vc), Gh = st({}, vc, { view: 0, detail: 0 }), dx = oi(Gh), qy, Yy, Xh, og = st({}, Gh, { screenX: 0, screenY: 0, clientX: 0, clientY: 0, pageX: 0, pageY: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, getModifierState: Qy, button: 0, buttons: 0, relatedTarget: function(i) {
    return i.relatedTarget === void 0 ? i.fromElement === i.srcElement ? i.toElement : i.fromElement : i.relatedTarget;
  }, movementX: function(i) {
    return "movementX" in i ? i.movementX : (i !== Xh && (Xh && i.type === "mousemove" ? (qy = i.screenX - Xh.screenX, Yy = i.screenY - Xh.screenY) : Yy = qy = 0, Xh = i), qy);
  }, movementY: function(i) {
    return "movementY" in i ? i.movementY : Yy;
  } }), iS = oi(og), fx = st({}, og, { dataTransfer: 0 }), px = oi(fx), gx = st({}, Gh, { relatedTarget: 0 }), Ky = oi(gx), mx = st({}, vc, { animationName: 0, elapsedTime: 0, pseudoElement: 0 }), yx = oi(mx), vx = st({}, vc, { clipboardData: function(i) {
    return "clipboardData" in i ? i.clipboardData : window.clipboardData;
  } }), wx = oi(vx), Ax = st({}, vc, { data: 0 }), sS = oi(Ax), Sx = {
    Esc: "Escape",
    Spacebar: " ",
    Left: "ArrowLeft",
    Up: "ArrowUp",
    Right: "ArrowRight",
    Down: "ArrowDown",
    Del: "Delete",
    Win: "OS",
    Menu: "ContextMenu",
    Apps: "ContextMenu",
    Scroll: "ScrollLock",
    MozPrintableKey: "Unidentified"
  }, bx = {
    8: "Backspace",
    9: "Tab",
    12: "Clear",
    13: "Enter",
    16: "Shift",
    17: "Control",
    18: "Alt",
    19: "Pause",
    20: "CapsLock",
    27: "Escape",
    32: " ",
    33: "PageUp",
    34: "PageDown",
    35: "End",
    36: "Home",
    37: "ArrowLeft",
    38: "ArrowUp",
    39: "ArrowRight",
    40: "ArrowDown",
    45: "Insert",
    46: "Delete",
    112: "F1",
    113: "F2",
    114: "F3",
    115: "F4",
    116: "F5",
    117: "F6",
    118: "F7",
    119: "F8",
    120: "F9",
    121: "F10",
    122: "F11",
    123: "F12",
    144: "NumLock",
    145: "ScrollLock",
    224: "Meta"
  }, Ex = { Alt: "altKey", Control: "ctrlKey", Meta: "metaKey", Shift: "shiftKey" };
  function _x(i) {
    var r = this.nativeEvent;
    return r.getModifierState ? r.getModifierState(i) : (i = Ex[i]) ? !!r[i] : !1;
  }
  function Qy() {
    return _x;
  }
  var Cx = st({}, Gh, { key: function(i) {
    if (i.key) {
      var r = Sx[i.key] || i.key;
      if (r !== "Unidentified") return r;
    }
    return i.type === "keypress" ? (i = sg(i), i === 13 ? "Enter" : String.fromCharCode(i)) : i.type === "keydown" || i.type === "keyup" ? bx[i.keyCode] || "Unidentified" : "";
  }, code: 0, location: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, repeat: 0, locale: 0, getModifierState: Qy, charCode: function(i) {
    return i.type === "keypress" ? sg(i) : 0;
  }, keyCode: function(i) {
    return i.type === "keydown" || i.type === "keyup" ? i.keyCode : 0;
  }, which: function(i) {
    return i.type === "keypress" ? sg(i) : i.type === "keydown" || i.type === "keyup" ? i.keyCode : 0;
  } }), xx = oi(Cx), Tx = st({}, og, { pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0 }), rS = oi(Tx), kx = st({}, Gh, { touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: Qy }), Px = oi(kx), Rx = st({}, vc, { propertyName: 0, elapsedTime: 0, pseudoElement: 0 }), Mx = oi(Rx), Lx = st({}, og, {
    deltaX: function(i) {
      return "deltaX" in i ? i.deltaX : "wheelDeltaX" in i ? -i.wheelDeltaX : 0;
    },
    deltaY: function(i) {
      return "deltaY" in i ? i.deltaY : "wheelDeltaY" in i ? -i.wheelDeltaY : "wheelDelta" in i ? -i.wheelDelta : 0;
    },
    deltaZ: 0,
    deltaMode: 0
  }), Dx = oi(Lx), Ix = [9, 13, 27, 32], Zy = u && "CompositionEvent" in window, qh = null;
  u && "documentMode" in document && (qh = document.documentMode);
  var Fx = u && "TextEvent" in window && !qh, oS = u && (!Zy || qh && 8 < qh && 11 >= qh), aS = " ", lS = !1;
  function cS(i, r) {
    switch (i) {
      case "keyup":
        return Ix.indexOf(r.keyCode) !== -1;
      case "keydown":
        return r.keyCode !== 229;
      case "keypress":
      case "mousedown":
      case "focusout":
        return !0;
      default:
        return !1;
    }
  }
  function uS(i) {
    return i = i.detail, typeof i == "object" && "data" in i ? i.data : null;
  }
  var wc = !1;
  function Nx(i, r) {
    switch (i) {
      case "compositionend":
        return uS(r);
      case "keypress":
        return r.which !== 32 ? null : (lS = !0, aS);
      case "textInput":
        return i = r.data, i === aS && lS ? null : i;
      default:
        return null;
    }
  }
  function Ox(i, r) {
    if (wc) return i === "compositionend" || !Zy && cS(i, r) ? (i = eS(), ig = Gy = yo = null, wc = !1, i) : null;
    switch (i) {
      case "paste":
        return null;
      case "keypress":
        if (!(r.ctrlKey || r.altKey || r.metaKey) || r.ctrlKey && r.altKey) {
          if (r.char && 1 < r.char.length) return r.char;
          if (r.which) return String.fromCharCode(r.which);
        }
        return null;
      case "compositionend":
        return oS && r.locale !== "ko" ? null : r.data;
      default:
        return null;
    }
  }
  var Bx = { color: !0, date: !0, datetime: !0, "datetime-local": !0, email: !0, month: !0, number: !0, password: !0, range: !0, search: !0, tel: !0, text: !0, time: !0, url: !0, week: !0 };
  function hS(i) {
    var r = i && i.nodeName && i.nodeName.toLowerCase();
    return r === "input" ? !!Bx[i.type] : r === "textarea";
  }
  function dS(i, r, c, d) {
    LA(d), r = hg(r, "onChange"), 0 < r.length && (c = new Xy("onChange", "change", null, c, d), i.push({ event: c, listeners: r }));
  }
  var Yh = null, Kh = null;
  function zx(i) {
    RS(i, 0);
  }
  function ag(i) {
    var r = _c(i);
    if (Di(r)) return i;
  }
  function Ux(i, r) {
    if (i === "change") return r;
  }
  var fS = !1;
  if (u) {
    var Jy;
    if (u) {
      var tv = "oninput" in document;
      if (!tv) {
        var pS = document.createElement("div");
        pS.setAttribute("oninput", "return;"), tv = typeof pS.oninput == "function";
      }
      Jy = tv;
    } else Jy = !1;
    fS = Jy && (!document.documentMode || 9 < document.documentMode);
  }
  function gS() {
    Yh && (Yh.detachEvent("onpropertychange", mS), Kh = Yh = null);
  }
  function mS(i) {
    if (i.propertyName === "value" && ag(Kh)) {
      var r = [];
      dS(r, Kh, i, Dy(i)), NA(zx, r);
    }
  }
  function Hx(i, r, c) {
    i === "focusin" ? (gS(), Yh = r, Kh = c, Yh.attachEvent("onpropertychange", mS)) : i === "focusout" && gS();
  }
  function jx(i) {
    if (i === "selectionchange" || i === "keyup" || i === "keydown") return ag(Kh);
  }
  function $x(i, r) {
    if (i === "click") return ag(r);
  }
  function Vx(i, r) {
    if (i === "input" || i === "change") return ag(r);
  }
  function Wx(i, r) {
    return i === r && (i !== 0 || 1 / i === 1 / r) || i !== i && r !== r;
  }
  var as = typeof Object.is == "function" ? Object.is : Wx;
  function Qh(i, r) {
    if (as(i, r)) return !0;
    if (typeof i != "object" || i === null || typeof r != "object" || r === null) return !1;
    var c = Object.keys(i), d = Object.keys(r);
    if (c.length !== d.length) return !1;
    for (d = 0; d < c.length; d++) {
      var m = c[d];
      if (!h.call(r, m) || !as(i[m], r[m])) return !1;
    }
    return !0;
  }
  function yS(i) {
    for (; i && i.firstChild; ) i = i.firstChild;
    return i;
  }
  function vS(i, r) {
    var c = yS(i);
    i = 0;
    for (var d; c; ) {
      if (c.nodeType === 3) {
        if (d = i + c.textContent.length, i <= r && d >= r) return { node: c, offset: r - i };
        i = d;
      }
      t: {
        for (; c; ) {
          if (c.nextSibling) {
            c = c.nextSibling;
            break t;
          }
          c = c.parentNode;
        }
        c = void 0;
      }
      c = yS(c);
    }
  }
  function wS(i, r) {
    return i && r ? i === r ? !0 : i && i.nodeType === 3 ? !1 : r && r.nodeType === 3 ? wS(i, r.parentNode) : "contains" in i ? i.contains(r) : i.compareDocumentPosition ? !!(i.compareDocumentPosition(r) & 16) : !1 : !1;
  }
  function AS() {
    for (var i = window, r = Ht(); r instanceof i.HTMLIFrameElement; ) {
      try {
        var c = typeof r.contentWindow.location.href == "string";
      } catch {
        c = !1;
      }
      if (c) i = r.contentWindow;
      else break;
      r = Ht(i.document);
    }
    return r;
  }
  function ev(i) {
    var r = i && i.nodeName && i.nodeName.toLowerCase();
    return r && (r === "input" && (i.type === "text" || i.type === "search" || i.type === "tel" || i.type === "url" || i.type === "password") || r === "textarea" || i.contentEditable === "true");
  }
  function Gx(i) {
    var r = AS(), c = i.focusedElem, d = i.selectionRange;
    if (r !== c && c && c.ownerDocument && wS(c.ownerDocument.documentElement, c)) {
      if (d !== null && ev(c)) {
        if (r = d.start, i = d.end, i === void 0 && (i = r), "selectionStart" in c) c.selectionStart = r, c.selectionEnd = Math.min(i, c.value.length);
        else if (i = (r = c.ownerDocument || document) && r.defaultView || window, i.getSelection) {
          i = i.getSelection();
          var m = c.textContent.length, A = Math.min(d.start, m);
          d = d.end === void 0 ? A : Math.min(d.end, m), !i.extend && A > d && (m = d, d = A, A = m), m = vS(c, A);
          var C = vS(
            c,
            d
          );
          m && C && (i.rangeCount !== 1 || i.anchorNode !== m.node || i.anchorOffset !== m.offset || i.focusNode !== C.node || i.focusOffset !== C.offset) && (r = r.createRange(), r.setStart(m.node, m.offset), i.removeAllRanges(), A > d ? (i.addRange(r), i.extend(C.node, C.offset)) : (r.setEnd(C.node, C.offset), i.addRange(r)));
        }
      }
      for (r = [], i = c; i = i.parentNode; ) i.nodeType === 1 && r.push({ element: i, left: i.scrollLeft, top: i.scrollTop });
      for (typeof c.focus == "function" && c.focus(), c = 0; c < r.length; c++) i = r[c], i.element.scrollLeft = i.left, i.element.scrollTop = i.top;
    }
  }
  var Xx = u && "documentMode" in document && 11 >= document.documentMode, Ac = null, nv = null, Zh = null, iv = !1;
  function SS(i, r, c) {
    var d = c.window === c ? c.document : c.nodeType === 9 ? c : c.ownerDocument;
    iv || Ac == null || Ac !== Ht(d) || (d = Ac, "selectionStart" in d && ev(d) ? d = { start: d.selectionStart, end: d.selectionEnd } : (d = (d.ownerDocument && d.ownerDocument.defaultView || window).getSelection(), d = { anchorNode: d.anchorNode, anchorOffset: d.anchorOffset, focusNode: d.focusNode, focusOffset: d.focusOffset }), Zh && Qh(Zh, d) || (Zh = d, d = hg(nv, "onSelect"), 0 < d.length && (r = new Xy("onSelect", "select", null, r, c), i.push({ event: r, listeners: d }), r.target = Ac)));
  }
  function lg(i, r) {
    var c = {};
    return c[i.toLowerCase()] = r.toLowerCase(), c["Webkit" + i] = "webkit" + r, c["Moz" + i] = "moz" + r, c;
  }
  var Sc = { animationend: lg("Animation", "AnimationEnd"), animationiteration: lg("Animation", "AnimationIteration"), animationstart: lg("Animation", "AnimationStart"), transitionend: lg("Transition", "TransitionEnd") }, sv = {}, bS = {};
  u && (bS = document.createElement("div").style, "AnimationEvent" in window || (delete Sc.animationend.animation, delete Sc.animationiteration.animation, delete Sc.animationstart.animation), "TransitionEvent" in window || delete Sc.transitionend.transition);
  function cg(i) {
    if (sv[i]) return sv[i];
    if (!Sc[i]) return i;
    var r = Sc[i], c;
    for (c in r) if (r.hasOwnProperty(c) && c in bS) return sv[i] = r[c];
    return i;
  }
  var ES = cg("animationend"), _S = cg("animationiteration"), CS = cg("animationstart"), xS = cg("transitionend"), TS = /* @__PURE__ */ new Map(), kS = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
  function vo(i, r) {
    TS.set(i, r), a(r, [i]);
  }
  for (var rv = 0; rv < kS.length; rv++) {
    var ov = kS[rv], qx = ov.toLowerCase(), Yx = ov[0].toUpperCase() + ov.slice(1);
    vo(qx, "on" + Yx);
  }
  vo(ES, "onAnimationEnd"), vo(_S, "onAnimationIteration"), vo(CS, "onAnimationStart"), vo("dblclick", "onDoubleClick"), vo("focusin", "onFocus"), vo("focusout", "onBlur"), vo(xS, "onTransitionEnd"), l("onMouseEnter", ["mouseout", "mouseover"]), l("onMouseLeave", ["mouseout", "mouseover"]), l("onPointerEnter", ["pointerout", "pointerover"]), l("onPointerLeave", ["pointerout", "pointerover"]), a("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" ")), a("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" ")), a("onBeforeInput", ["compositionend", "keypress", "textInput", "paste"]), a("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" ")), a("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" ")), a("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
  var Jh = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), Kx = new Set("cancel close invalid load scroll toggle".split(" ").concat(Jh));
  function PS(i, r, c) {
    var d = i.type || "unknown-event";
    i.currentTarget = c, qC(d, r, void 0, i), i.currentTarget = null;
  }
  function RS(i, r) {
    r = (r & 4) !== 0;
    for (var c = 0; c < i.length; c++) {
      var d = i[c], m = d.event;
      d = d.listeners;
      t: {
        var A = void 0;
        if (r) for (var C = d.length - 1; 0 <= C; C--) {
          var M = d[C], F = M.instance, j = M.currentTarget;
          if (M = M.listener, F !== A && m.isPropagationStopped()) break t;
          PS(m, M, j), A = F;
        }
        else for (C = 0; C < d.length; C++) {
          if (M = d[C], F = M.instance, j = M.currentTarget, M = M.listener, F !== A && m.isPropagationStopped()) break t;
          PS(m, M, j), A = F;
        }
      }
    }
    if (Xp) throw i = Oy, Xp = !1, Oy = null, i;
  }
  function pe(i, r) {
    var c = r[pv];
    c === void 0 && (c = r[pv] = /* @__PURE__ */ new Set());
    var d = i + "__bubble";
    c.has(d) || (MS(r, i, 2, !1), c.add(d));
  }
  function av(i, r, c) {
    var d = 0;
    r && (d |= 4), MS(c, i, d, r);
  }
  var ug = "_reactListening" + Math.random().toString(36).slice(2);
  function td(i) {
    if (!i[ug]) {
      i[ug] = !0, n.forEach(function(c) {
        c !== "selectionchange" && (Kx.has(c) || av(c, !1, i), av(c, !0, i));
      });
      var r = i.nodeType === 9 ? i : i.ownerDocument;
      r === null || r[ug] || (r[ug] = !0, av("selectionchange", !1, r));
    }
  }
  function MS(i, r, c, d) {
    switch (tS(r)) {
      case 1:
        var m = ux;
        break;
      case 4:
        m = hx;
        break;
      default:
        m = Vy;
    }
    c = m.bind(null, r, c, i), m = void 0, !Ny || r !== "touchstart" && r !== "touchmove" && r !== "wheel" || (m = !0), d ? m !== void 0 ? i.addEventListener(r, c, { capture: !0, passive: m }) : i.addEventListener(r, c, !0) : m !== void 0 ? i.addEventListener(r, c, { passive: m }) : i.addEventListener(r, c, !1);
  }
  function lv(i, r, c, d, m) {
    var A = d;
    if ((r & 1) === 0 && (r & 2) === 0 && d !== null) t: for (; ; ) {
      if (d === null) return;
      var C = d.tag;
      if (C === 3 || C === 4) {
        var M = d.stateNode.containerInfo;
        if (M === m || M.nodeType === 8 && M.parentNode === m) break;
        if (C === 4) for (C = d.return; C !== null; ) {
          var F = C.tag;
          if ((F === 3 || F === 4) && (F = C.stateNode.containerInfo, F === m || F.nodeType === 8 && F.parentNode === m)) return;
          C = C.return;
        }
        for (; M !== null; ) {
          if (C = Ia(M), C === null) return;
          if (F = C.tag, F === 5 || F === 6) {
            d = A = C;
            continue t;
          }
          M = M.parentNode;
        }
      }
      d = d.return;
    }
    NA(function() {
      var j = A, Y = Dy(c), J = [];
      t: {
        var G = TS.get(i);
        if (G !== void 0) {
          var ut = Xy, mt = i;
          switch (i) {
            case "keypress":
              if (sg(c) === 0) break t;
            case "keydown":
            case "keyup":
              ut = xx;
              break;
            case "focusin":
              mt = "focus", ut = Ky;
              break;
            case "focusout":
              mt = "blur", ut = Ky;
              break;
            case "beforeblur":
            case "afterblur":
              ut = Ky;
              break;
            case "click":
              if (c.button === 2) break t;
            case "auxclick":
            case "dblclick":
            case "mousedown":
            case "mousemove":
            case "mouseup":
            case "mouseout":
            case "mouseover":
            case "contextmenu":
              ut = iS;
              break;
            case "drag":
            case "dragend":
            case "dragenter":
            case "dragexit":
            case "dragleave":
            case "dragover":
            case "dragstart":
            case "drop":
              ut = px;
              break;
            case "touchcancel":
            case "touchend":
            case "touchmove":
            case "touchstart":
              ut = Px;
              break;
            case ES:
            case _S:
            case CS:
              ut = yx;
              break;
            case xS:
              ut = Mx;
              break;
            case "scroll":
              ut = dx;
              break;
            case "wheel":
              ut = Dx;
              break;
            case "copy":
            case "cut":
            case "paste":
              ut = wx;
              break;
            case "gotpointercapture":
            case "lostpointercapture":
            case "pointercancel":
            case "pointerdown":
            case "pointermove":
            case "pointerout":
            case "pointerover":
            case "pointerup":
              ut = rS;
          }
          var yt = (r & 4) !== 0, Ne = !yt && i === "scroll", U = yt ? G !== null ? G + "Capture" : null : G;
          yt = [];
          for (var O = j, H; O !== null; ) {
            H = O;
            var et = H.stateNode;
            if (H.tag === 5 && et !== null && (H = et, U !== null && (et = Nh(O, U), et != null && yt.push(ed(O, et, H)))), Ne) break;
            O = O.return;
          }
          0 < yt.length && (G = new ut(G, mt, null, c, Y), J.push({ event: G, listeners: yt }));
        }
      }
      if ((r & 7) === 0) {
        t: {
          if (G = i === "mouseover" || i === "pointerover", ut = i === "mouseout" || i === "pointerout", G && c !== Ly && (mt = c.relatedTarget || c.fromElement) && (Ia(mt) || mt[dr])) break t;
          if ((ut || G) && (G = Y.window === Y ? Y : (G = Y.ownerDocument) ? G.defaultView || G.parentWindow : window, ut ? (mt = c.relatedTarget || c.toElement, ut = j, mt = mt ? Ia(mt) : null, mt !== null && (Ne = Da(mt), mt !== Ne || mt.tag !== 5 && mt.tag !== 6) && (mt = null)) : (ut = null, mt = j), ut !== mt)) {
            if (yt = iS, et = "onMouseLeave", U = "onMouseEnter", O = "mouse", (i === "pointerout" || i === "pointerover") && (yt = rS, et = "onPointerLeave", U = "onPointerEnter", O = "pointer"), Ne = ut == null ? G : _c(ut), H = mt == null ? G : _c(mt), G = new yt(et, O + "leave", ut, c, Y), G.target = Ne, G.relatedTarget = H, et = null, Ia(Y) === j && (yt = new yt(U, O + "enter", mt, c, Y), yt.target = H, yt.relatedTarget = Ne, et = yt), Ne = et, ut && mt) e: {
              for (yt = ut, U = mt, O = 0, H = yt; H; H = bc(H)) O++;
              for (H = 0, et = U; et; et = bc(et)) H++;
              for (; 0 < O - H; ) yt = bc(yt), O--;
              for (; 0 < H - O; ) U = bc(U), H--;
              for (; O--; ) {
                if (yt === U || U !== null && yt === U.alternate) break e;
                yt = bc(yt), U = bc(U);
              }
              yt = null;
            }
            else yt = null;
            ut !== null && LS(J, G, ut, yt, !1), mt !== null && Ne !== null && LS(J, Ne, mt, yt, !0);
          }
        }
        t: {
          if (G = j ? _c(j) : window, ut = G.nodeName && G.nodeName.toLowerCase(), ut === "select" || ut === "input" && G.type === "file") var At = Ux;
          else if (hS(G)) if (fS) At = Vx;
          else {
            At = jx;
            var _t = Hx;
          }
          else (ut = G.nodeName) && ut.toLowerCase() === "input" && (G.type === "checkbox" || G.type === "radio") && (At = $x);
          if (At && (At = At(i, j))) {
            dS(J, At, c, Y);
            break t;
          }
          _t && _t(i, G, j), i === "focusout" && (_t = G._wrapperState) && _t.controlled && G.type === "number" && Ma(G, "number", G.value);
        }
        switch (_t = j ? _c(j) : window, i) {
          case "focusin":
            (hS(_t) || _t.contentEditable === "true") && (Ac = _t, nv = j, Zh = null);
            break;
          case "focusout":
            Zh = nv = Ac = null;
            break;
          case "mousedown":
            iv = !0;
            break;
          case "contextmenu":
          case "mouseup":
          case "dragend":
            iv = !1, SS(J, c, Y);
            break;
          case "selectionchange":
            if (Xx) break;
          case "keydown":
          case "keyup":
            SS(J, c, Y);
        }
        var Ct;
        if (Zy) t: {
          switch (i) {
            case "compositionstart":
              var kt = "onCompositionStart";
              break t;
            case "compositionend":
              kt = "onCompositionEnd";
              break t;
            case "compositionupdate":
              kt = "onCompositionUpdate";
              break t;
          }
          kt = void 0;
        }
        else wc ? cS(i, c) && (kt = "onCompositionEnd") : i === "keydown" && c.keyCode === 229 && (kt = "onCompositionStart");
        kt && (oS && c.locale !== "ko" && (wc || kt !== "onCompositionStart" ? kt === "onCompositionEnd" && wc && (Ct = eS()) : (yo = Y, Gy = "value" in yo ? yo.value : yo.textContent, wc = !0)), _t = hg(j, kt), 0 < _t.length && (kt = new sS(kt, i, null, c, Y), J.push({ event: kt, listeners: _t }), Ct ? kt.data = Ct : (Ct = uS(c), Ct !== null && (kt.data = Ct)))), (Ct = Fx ? Nx(i, c) : Ox(i, c)) && (j = hg(j, "onBeforeInput"), 0 < j.length && (Y = new sS("onBeforeInput", "beforeinput", null, c, Y), J.push({ event: Y, listeners: j }), Y.data = Ct));
      }
      RS(J, r);
    });
  }
  function ed(i, r, c) {
    return { instance: i, listener: r, currentTarget: c };
  }
  function hg(i, r) {
    for (var c = r + "Capture", d = []; i !== null; ) {
      var m = i, A = m.stateNode;
      m.tag === 5 && A !== null && (m = A, A = Nh(i, c), A != null && d.unshift(ed(i, A, m)), A = Nh(i, r), A != null && d.push(ed(i, A, m))), i = i.return;
    }
    return d;
  }
  function bc(i) {
    if (i === null) return null;
    do
      i = i.return;
    while (i && i.tag !== 5);
    return i || null;
  }
  function LS(i, r, c, d, m) {
    for (var A = r._reactName, C = []; c !== null && c !== d; ) {
      var M = c, F = M.alternate, j = M.stateNode;
      if (F !== null && F === d) break;
      M.tag === 5 && j !== null && (M = j, m ? (F = Nh(c, A), F != null && C.unshift(ed(c, F, M))) : m || (F = Nh(c, A), F != null && C.push(ed(c, F, M)))), c = c.return;
    }
    C.length !== 0 && i.push({ event: r, listeners: C });
  }
  var Qx = /\r\n?/g, Zx = /\u0000|\uFFFD/g;
  function DS(i) {
    return (typeof i == "string" ? i : "" + i).replace(Qx, `
`).replace(Zx, "");
  }
  function dg(i, r, c) {
    if (r = DS(r), DS(i) !== r && c) throw Error(e(425));
  }
  function fg() {
  }
  var cv = null, uv = null;
  function hv(i, r) {
    return i === "textarea" || i === "noscript" || typeof r.children == "string" || typeof r.children == "number" || typeof r.dangerouslySetInnerHTML == "object" && r.dangerouslySetInnerHTML !== null && r.dangerouslySetInnerHTML.__html != null;
  }
  var dv = typeof setTimeout == "function" ? setTimeout : void 0, Jx = typeof clearTimeout == "function" ? clearTimeout : void 0, IS = typeof Promise == "function" ? Promise : void 0, tT = typeof queueMicrotask == "function" ? queueMicrotask : typeof IS < "u" ? function(i) {
    return IS.resolve(null).then(i).catch(eT);
  } : dv;
  function eT(i) {
    setTimeout(function() {
      throw i;
    });
  }
  function fv(i, r) {
    var c = r, d = 0;
    do {
      var m = c.nextSibling;
      if (i.removeChild(c), m && m.nodeType === 8) if (c = m.data, c === "/$") {
        if (d === 0) {
          i.removeChild(m), Wh(r);
          return;
        }
        d--;
      } else c !== "$" && c !== "$?" && c !== "$!" || d++;
      c = m;
    } while (c);
    Wh(r);
  }
  function wo(i) {
    for (; i != null; i = i.nextSibling) {
      var r = i.nodeType;
      if (r === 1 || r === 3) break;
      if (r === 8) {
        if (r = i.data, r === "$" || r === "$!" || r === "$?") break;
        if (r === "/$") return null;
      }
    }
    return i;
  }
  function FS(i) {
    i = i.previousSibling;
    for (var r = 0; i; ) {
      if (i.nodeType === 8) {
        var c = i.data;
        if (c === "$" || c === "$!" || c === "$?") {
          if (r === 0) return i;
          r--;
        } else c === "/$" && r++;
      }
      i = i.previousSibling;
    }
    return null;
  }
  var Ec = Math.random().toString(36).slice(2), Ls = "__reactFiber$" + Ec, nd = "__reactProps$" + Ec, dr = "__reactContainer$" + Ec, pv = "__reactEvents$" + Ec, nT = "__reactListeners$" + Ec, iT = "__reactHandles$" + Ec;
  function Ia(i) {
    var r = i[Ls];
    if (r) return r;
    for (var c = i.parentNode; c; ) {
      if (r = c[dr] || c[Ls]) {
        if (c = r.alternate, r.child !== null || c !== null && c.child !== null) for (i = FS(i); i !== null; ) {
          if (c = i[Ls]) return c;
          i = FS(i);
        }
        return r;
      }
      i = c, c = i.parentNode;
    }
    return null;
  }
  function id(i) {
    return i = i[Ls] || i[dr], !i || i.tag !== 5 && i.tag !== 6 && i.tag !== 13 && i.tag !== 3 ? null : i;
  }
  function _c(i) {
    if (i.tag === 5 || i.tag === 6) return i.stateNode;
    throw Error(e(33));
  }
  function pg(i) {
    return i[nd] || null;
  }
  var gv = [], Cc = -1;
  function Ao(i) {
    return { current: i };
  }
  function ge(i) {
    0 > Cc || (i.current = gv[Cc], gv[Cc] = null, Cc--);
  }
  function ue(i, r) {
    Cc++, gv[Cc] = i.current, i.current = r;
  }
  var So = {}, Sn = Ao(So), Vn = Ao(!1), Fa = So;
  function xc(i, r) {
    var c = i.type.contextTypes;
    if (!c) return So;
    var d = i.stateNode;
    if (d && d.__reactInternalMemoizedUnmaskedChildContext === r) return d.__reactInternalMemoizedMaskedChildContext;
    var m = {}, A;
    for (A in c) m[A] = r[A];
    return d && (i = i.stateNode, i.__reactInternalMemoizedUnmaskedChildContext = r, i.__reactInternalMemoizedMaskedChildContext = m), m;
  }
  function Wn(i) {
    return i = i.childContextTypes, i != null;
  }
  function gg() {
    ge(Vn), ge(Sn);
  }
  function NS(i, r, c) {
    if (Sn.current !== So) throw Error(e(168));
    ue(Sn, r), ue(Vn, c);
  }
  function OS(i, r, c) {
    var d = i.stateNode;
    if (r = r.childContextTypes, typeof d.getChildContext != "function") return c;
    d = d.getChildContext();
    for (var m in d) if (!(m in r)) throw Error(e(108, bt(i) || "Unknown", m));
    return st({}, c, d);
  }
  function mg(i) {
    return i = (i = i.stateNode) && i.__reactInternalMemoizedMergedChildContext || So, Fa = Sn.current, ue(Sn, i), ue(Vn, Vn.current), !0;
  }
  function BS(i, r, c) {
    var d = i.stateNode;
    if (!d) throw Error(e(169));
    c ? (i = OS(i, r, Fa), d.__reactInternalMemoizedMergedChildContext = i, ge(Vn), ge(Sn), ue(Sn, i)) : ge(Vn), ue(Vn, c);
  }
  var fr = null, yg = !1, mv = !1;
  function zS(i) {
    fr === null ? fr = [i] : fr.push(i);
  }
  function sT(i) {
    yg = !0, zS(i);
  }
  function bo() {
    if (!mv && fr !== null) {
      mv = !0;
      var i = 0, r = Kt;
      try {
        var c = fr;
        for (Kt = 1; i < c.length; i++) {
          var d = c[i];
          do
            d = d(!0);
          while (d !== null);
        }
        fr = null, yg = !1;
      } catch (m) {
        throw fr !== null && (fr = fr.slice(i + 1)), HA(By, bo), m;
      } finally {
        Kt = r, mv = !1;
      }
    }
    return null;
  }
  var Tc = [], kc = 0, vg = null, wg = 0, Fi = [], Ni = 0, Na = null, pr = 1, gr = "";
  function Oa(i, r) {
    Tc[kc++] = wg, Tc[kc++] = vg, vg = i, wg = r;
  }
  function US(i, r, c) {
    Fi[Ni++] = pr, Fi[Ni++] = gr, Fi[Ni++] = Na, Na = i;
    var d = pr;
    i = gr;
    var m = 32 - os(d) - 1;
    d &= ~(1 << m), c += 1;
    var A = 32 - os(r) + m;
    if (30 < A) {
      var C = m - m % 5;
      A = (d & (1 << C) - 1).toString(32), d >>= C, m -= C, pr = 1 << 32 - os(r) + m | c << m | d, gr = A + i;
    } else pr = 1 << A | c << m | d, gr = i;
  }
  function yv(i) {
    i.return !== null && (Oa(i, 1), US(i, 1, 0));
  }
  function vv(i) {
    for (; i === vg; ) vg = Tc[--kc], Tc[kc] = null, wg = Tc[--kc], Tc[kc] = null;
    for (; i === Na; ) Na = Fi[--Ni], Fi[Ni] = null, gr = Fi[--Ni], Fi[Ni] = null, pr = Fi[--Ni], Fi[Ni] = null;
  }
  var ai = null, li = null, ye = !1, ls = null;
  function HS(i, r) {
    var c = Ui(5, null, null, 0);
    c.elementType = "DELETED", c.stateNode = r, c.return = i, r = i.deletions, r === null ? (i.deletions = [c], i.flags |= 16) : r.push(c);
  }
  function jS(i, r) {
    switch (i.tag) {
      case 5:
        var c = i.type;
        return r = r.nodeType !== 1 || c.toLowerCase() !== r.nodeName.toLowerCase() ? null : r, r !== null ? (i.stateNode = r, ai = i, li = wo(r.firstChild), !0) : !1;
      case 6:
        return r = i.pendingProps === "" || r.nodeType !== 3 ? null : r, r !== null ? (i.stateNode = r, ai = i, li = null, !0) : !1;
      case 13:
        return r = r.nodeType !== 8 ? null : r, r !== null ? (c = Na !== null ? { id: pr, overflow: gr } : null, i.memoizedState = { dehydrated: r, treeContext: c, retryLane: 1073741824 }, c = Ui(18, null, null, 0), c.stateNode = r, c.return = i, i.child = c, ai = i, li = null, !0) : !1;
      default:
        return !1;
    }
  }
  function wv(i) {
    return (i.mode & 1) !== 0 && (i.flags & 128) === 0;
  }
  function Av(i) {
    if (ye) {
      var r = li;
      if (r) {
        var c = r;
        if (!jS(i, r)) {
          if (wv(i)) throw Error(e(418));
          r = wo(c.nextSibling);
          var d = ai;
          r && jS(i, r) ? HS(d, c) : (i.flags = i.flags & -4097 | 2, ye = !1, ai = i);
        }
      } else {
        if (wv(i)) throw Error(e(418));
        i.flags = i.flags & -4097 | 2, ye = !1, ai = i;
      }
    }
  }
  function $S(i) {
    for (i = i.return; i !== null && i.tag !== 5 && i.tag !== 3 && i.tag !== 13; ) i = i.return;
    ai = i;
  }
  function Ag(i) {
    if (i !== ai) return !1;
    if (!ye) return $S(i), ye = !0, !1;
    var r;
    if ((r = i.tag !== 3) && !(r = i.tag !== 5) && (r = i.type, r = r !== "head" && r !== "body" && !hv(i.type, i.memoizedProps)), r && (r = li)) {
      if (wv(i)) throw VS(), Error(e(418));
      for (; r; ) HS(i, r), r = wo(r.nextSibling);
    }
    if ($S(i), i.tag === 13) {
      if (i = i.memoizedState, i = i !== null ? i.dehydrated : null, !i) throw Error(e(317));
      t: {
        for (i = i.nextSibling, r = 0; i; ) {
          if (i.nodeType === 8) {
            var c = i.data;
            if (c === "/$") {
              if (r === 0) {
                li = wo(i.nextSibling);
                break t;
              }
              r--;
            } else c !== "$" && c !== "$!" && c !== "$?" || r++;
          }
          i = i.nextSibling;
        }
        li = null;
      }
    } else li = ai ? wo(i.stateNode.nextSibling) : null;
    return !0;
  }
  function VS() {
    for (var i = li; i; ) i = wo(i.nextSibling);
  }
  function Pc() {
    li = ai = null, ye = !1;
  }
  function Sv(i) {
    ls === null ? ls = [i] : ls.push(i);
  }
  var rT = L.ReactCurrentBatchConfig;
  function sd(i, r, c) {
    if (i = c.ref, i !== null && typeof i != "function" && typeof i != "object") {
      if (c._owner) {
        if (c = c._owner, c) {
          if (c.tag !== 1) throw Error(e(309));
          var d = c.stateNode;
        }
        if (!d) throw Error(e(147, i));
        var m = d, A = "" + i;
        return r !== null && r.ref !== null && typeof r.ref == "function" && r.ref._stringRef === A ? r.ref : (r = function(C) {
          var M = m.refs;
          C === null ? delete M[A] : M[A] = C;
        }, r._stringRef = A, r);
      }
      if (typeof i != "string") throw Error(e(284));
      if (!c._owner) throw Error(e(290, i));
    }
    return i;
  }
  function Sg(i, r) {
    throw i = Object.prototype.toString.call(r), Error(e(31, i === "[object Object]" ? "object with keys {" + Object.keys(r).join(", ") + "}" : i));
  }
  function WS(i) {
    var r = i._init;
    return r(i._payload);
  }
  function GS(i) {
    function r(U, O) {
      if (i) {
        var H = U.deletions;
        H === null ? (U.deletions = [O], U.flags |= 16) : H.push(O);
      }
    }
    function c(U, O) {
      if (!i) return null;
      for (; O !== null; ) r(U, O), O = O.sibling;
      return null;
    }
    function d(U, O) {
      for (U = /* @__PURE__ */ new Map(); O !== null; ) O.key !== null ? U.set(O.key, O) : U.set(O.index, O), O = O.sibling;
      return U;
    }
    function m(U, O) {
      return U = Ro(U, O), U.index = 0, U.sibling = null, U;
    }
    function A(U, O, H) {
      return U.index = H, i ? (H = U.alternate, H !== null ? (H = H.index, H < O ? (U.flags |= 2, O) : H) : (U.flags |= 2, O)) : (U.flags |= 1048576, O);
    }
    function C(U) {
      return i && U.alternate === null && (U.flags |= 2), U;
    }
    function M(U, O, H, et) {
      return O === null || O.tag !== 6 ? (O = d0(H, U.mode, et), O.return = U, O) : (O = m(O, H), O.return = U, O);
    }
    function F(U, O, H, et) {
      var At = H.type;
      return At === B ? Y(U, O, H.props.children, et, H.key) : O !== null && (O.elementType === At || typeof At == "object" && At !== null && At.$$typeof === tt && WS(At) === O.type) ? (et = m(O, H.props), et.ref = sd(U, O, H), et.return = U, et) : (et = Wg(H.type, H.key, H.props, null, U.mode, et), et.ref = sd(U, O, H), et.return = U, et);
    }
    function j(U, O, H, et) {
      return O === null || O.tag !== 4 || O.stateNode.containerInfo !== H.containerInfo || O.stateNode.implementation !== H.implementation ? (O = f0(H, U.mode, et), O.return = U, O) : (O = m(O, H.children || []), O.return = U, O);
    }
    function Y(U, O, H, et, At) {
      return O === null || O.tag !== 7 ? (O = Wa(H, U.mode, et, At), O.return = U, O) : (O = m(O, H), O.return = U, O);
    }
    function J(U, O, H) {
      if (typeof O == "string" && O !== "" || typeof O == "number") return O = d0("" + O, U.mode, H), O.return = U, O;
      if (typeof O == "object" && O !== null) {
        switch (O.$$typeof) {
          case D:
            return H = Wg(O.type, O.key, O.props, null, U.mode, H), H.ref = sd(U, null, O), H.return = U, H;
          case N:
            return O = f0(O, U.mode, H), O.return = U, O;
          case tt:
            var et = O._init;
            return J(U, et(O._payload), H);
        }
        if (La(O) || ot(O)) return O = Wa(O, U.mode, H, null), O.return = U, O;
        Sg(U, O);
      }
      return null;
    }
    function G(U, O, H, et) {
      var At = O !== null ? O.key : null;
      if (typeof H == "string" && H !== "" || typeof H == "number") return At !== null ? null : M(U, O, "" + H, et);
      if (typeof H == "object" && H !== null) {
        switch (H.$$typeof) {
          case D:
            return H.key === At ? F(U, O, H, et) : null;
          case N:
            return H.key === At ? j(U, O, H, et) : null;
          case tt:
            return At = H._init, G(
              U,
              O,
              At(H._payload),
              et
            );
        }
        if (La(H) || ot(H)) return At !== null ? null : Y(U, O, H, et, null);
        Sg(U, H);
      }
      return null;
    }
    function ut(U, O, H, et, At) {
      if (typeof et == "string" && et !== "" || typeof et == "number") return U = U.get(H) || null, M(O, U, "" + et, At);
      if (typeof et == "object" && et !== null) {
        switch (et.$$typeof) {
          case D:
            return U = U.get(et.key === null ? H : et.key) || null, F(O, U, et, At);
          case N:
            return U = U.get(et.key === null ? H : et.key) || null, j(O, U, et, At);
          case tt:
            var _t = et._init;
            return ut(U, O, H, _t(et._payload), At);
        }
        if (La(et) || ot(et)) return U = U.get(H) || null, Y(O, U, et, At, null);
        Sg(O, et);
      }
      return null;
    }
    function mt(U, O, H, et) {
      for (var At = null, _t = null, Ct = O, kt = O = 0, tn = null; Ct !== null && kt < H.length; kt++) {
        Ct.index > kt ? (tn = Ct, Ct = null) : tn = Ct.sibling;
        var Xt = G(U, Ct, H[kt], et);
        if (Xt === null) {
          Ct === null && (Ct = tn);
          break;
        }
        i && Ct && Xt.alternate === null && r(U, Ct), O = A(Xt, O, kt), _t === null ? At = Xt : _t.sibling = Xt, _t = Xt, Ct = tn;
      }
      if (kt === H.length) return c(U, Ct), ye && Oa(U, kt), At;
      if (Ct === null) {
        for (; kt < H.length; kt++) Ct = J(U, H[kt], et), Ct !== null && (O = A(Ct, O, kt), _t === null ? At = Ct : _t.sibling = Ct, _t = Ct);
        return ye && Oa(U, kt), At;
      }
      for (Ct = d(U, Ct); kt < H.length; kt++) tn = ut(Ct, U, kt, H[kt], et), tn !== null && (i && tn.alternate !== null && Ct.delete(tn.key === null ? kt : tn.key), O = A(tn, O, kt), _t === null ? At = tn : _t.sibling = tn, _t = tn);
      return i && Ct.forEach(function(Mo) {
        return r(U, Mo);
      }), ye && Oa(U, kt), At;
    }
    function yt(U, O, H, et) {
      var At = ot(H);
      if (typeof At != "function") throw Error(e(150));
      if (H = At.call(H), H == null) throw Error(e(151));
      for (var _t = At = null, Ct = O, kt = O = 0, tn = null, Xt = H.next(); Ct !== null && !Xt.done; kt++, Xt = H.next()) {
        Ct.index > kt ? (tn = Ct, Ct = null) : tn = Ct.sibling;
        var Mo = G(U, Ct, Xt.value, et);
        if (Mo === null) {
          Ct === null && (Ct = tn);
          break;
        }
        i && Ct && Mo.alternate === null && r(U, Ct), O = A(Mo, O, kt), _t === null ? At = Mo : _t.sibling = Mo, _t = Mo, Ct = tn;
      }
      if (Xt.done) return c(
        U,
        Ct
      ), ye && Oa(U, kt), At;
      if (Ct === null) {
        for (; !Xt.done; kt++, Xt = H.next()) Xt = J(U, Xt.value, et), Xt !== null && (O = A(Xt, O, kt), _t === null ? At = Xt : _t.sibling = Xt, _t = Xt);
        return ye && Oa(U, kt), At;
      }
      for (Ct = d(U, Ct); !Xt.done; kt++, Xt = H.next()) Xt = ut(Ct, U, kt, Xt.value, et), Xt !== null && (i && Xt.alternate !== null && Ct.delete(Xt.key === null ? kt : Xt.key), O = A(Xt, O, kt), _t === null ? At = Xt : _t.sibling = Xt, _t = Xt);
      return i && Ct.forEach(function(BT) {
        return r(U, BT);
      }), ye && Oa(U, kt), At;
    }
    function Ne(U, O, H, et) {
      if (typeof H == "object" && H !== null && H.type === B && H.key === null && (H = H.props.children), typeof H == "object" && H !== null) {
        switch (H.$$typeof) {
          case D:
            t: {
              for (var At = H.key, _t = O; _t !== null; ) {
                if (_t.key === At) {
                  if (At = H.type, At === B) {
                    if (_t.tag === 7) {
                      c(U, _t.sibling), O = m(_t, H.props.children), O.return = U, U = O;
                      break t;
                    }
                  } else if (_t.elementType === At || typeof At == "object" && At !== null && At.$$typeof === tt && WS(At) === _t.type) {
                    c(U, _t.sibling), O = m(_t, H.props), O.ref = sd(U, _t, H), O.return = U, U = O;
                    break t;
                  }
                  c(U, _t);
                  break;
                } else r(U, _t);
                _t = _t.sibling;
              }
              H.type === B ? (O = Wa(H.props.children, U.mode, et, H.key), O.return = U, U = O) : (et = Wg(H.type, H.key, H.props, null, U.mode, et), et.ref = sd(U, O, H), et.return = U, U = et);
            }
            return C(U);
          case N:
            t: {
              for (_t = H.key; O !== null; ) {
                if (O.key === _t) if (O.tag === 4 && O.stateNode.containerInfo === H.containerInfo && O.stateNode.implementation === H.implementation) {
                  c(U, O.sibling), O = m(O, H.children || []), O.return = U, U = O;
                  break t;
                } else {
                  c(U, O);
                  break;
                }
                else r(U, O);
                O = O.sibling;
              }
              O = f0(H, U.mode, et), O.return = U, U = O;
            }
            return C(U);
          case tt:
            return _t = H._init, Ne(U, O, _t(H._payload), et);
        }
        if (La(H)) return mt(U, O, H, et);
        if (ot(H)) return yt(U, O, H, et);
        Sg(U, H);
      }
      return typeof H == "string" && H !== "" || typeof H == "number" ? (H = "" + H, O !== null && O.tag === 6 ? (c(U, O.sibling), O = m(O, H), O.return = U, U = O) : (c(U, O), O = d0(H, U.mode, et), O.return = U, U = O), C(U)) : c(U, O);
    }
    return Ne;
  }
  var Rc = GS(!0), XS = GS(!1), bg = Ao(null), Eg = null, Mc = null, bv = null;
  function Ev() {
    bv = Mc = Eg = null;
  }
  function _v(i) {
    var r = bg.current;
    ge(bg), i._currentValue = r;
  }
  function Cv(i, r, c) {
    for (; i !== null; ) {
      var d = i.alternate;
      if ((i.childLanes & r) !== r ? (i.childLanes |= r, d !== null && (d.childLanes |= r)) : d !== null && (d.childLanes & r) !== r && (d.childLanes |= r), i === c) break;
      i = i.return;
    }
  }
  function Lc(i, r) {
    Eg = i, bv = Mc = null, i = i.dependencies, i !== null && i.firstContext !== null && ((i.lanes & r) !== 0 && (Gn = !0), i.firstContext = null);
  }
  function Oi(i) {
    var r = i._currentValue;
    if (bv !== i) if (i = { context: i, memoizedValue: r, next: null }, Mc === null) {
      if (Eg === null) throw Error(e(308));
      Mc = i, Eg.dependencies = { lanes: 0, firstContext: i };
    } else Mc = Mc.next = i;
    return r;
  }
  var Ba = null;
  function xv(i) {
    Ba === null ? Ba = [i] : Ba.push(i);
  }
  function qS(i, r, c, d) {
    var m = r.interleaved;
    return m === null ? (c.next = c, xv(r)) : (c.next = m.next, m.next = c), r.interleaved = c, mr(i, d);
  }
  function mr(i, r) {
    i.lanes |= r;
    var c = i.alternate;
    for (c !== null && (c.lanes |= r), c = i, i = i.return; i !== null; ) i.childLanes |= r, c = i.alternate, c !== null && (c.childLanes |= r), c = i, i = i.return;
    return c.tag === 3 ? c.stateNode : null;
  }
  var Eo = !1;
  function Tv(i) {
    i.updateQueue = { baseState: i.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: { pending: null, interleaved: null, lanes: 0 }, effects: null };
  }
  function YS(i, r) {
    i = i.updateQueue, r.updateQueue === i && (r.updateQueue = { baseState: i.baseState, firstBaseUpdate: i.firstBaseUpdate, lastBaseUpdate: i.lastBaseUpdate, shared: i.shared, effects: i.effects });
  }
  function yr(i, r) {
    return { eventTime: i, lane: r, tag: 0, payload: null, callback: null, next: null };
  }
  function _o(i, r, c) {
    var d = i.updateQueue;
    if (d === null) return null;
    if (d = d.shared, (Vt & 2) !== 0) {
      var m = d.pending;
      return m === null ? r.next = r : (r.next = m.next, m.next = r), d.pending = r, mr(i, c);
    }
    return m = d.interleaved, m === null ? (r.next = r, xv(d)) : (r.next = m.next, m.next = r), d.interleaved = r, mr(i, c);
  }
  function _g(i, r, c) {
    if (r = r.updateQueue, r !== null && (r = r.shared, (c & 4194240) !== 0)) {
      var d = r.lanes;
      d &= i.pendingLanes, c |= d, r.lanes = c, Hy(i, c);
    }
  }
  function KS(i, r) {
    var c = i.updateQueue, d = i.alternate;
    if (d !== null && (d = d.updateQueue, c === d)) {
      var m = null, A = null;
      if (c = c.firstBaseUpdate, c !== null) {
        do {
          var C = { eventTime: c.eventTime, lane: c.lane, tag: c.tag, payload: c.payload, callback: c.callback, next: null };
          A === null ? m = A = C : A = A.next = C, c = c.next;
        } while (c !== null);
        A === null ? m = A = r : A = A.next = r;
      } else m = A = r;
      c = { baseState: d.baseState, firstBaseUpdate: m, lastBaseUpdate: A, shared: d.shared, effects: d.effects }, i.updateQueue = c;
      return;
    }
    i = c.lastBaseUpdate, i === null ? c.firstBaseUpdate = r : i.next = r, c.lastBaseUpdate = r;
  }
  function Cg(i, r, c, d) {
    var m = i.updateQueue;
    Eo = !1;
    var A = m.firstBaseUpdate, C = m.lastBaseUpdate, M = m.shared.pending;
    if (M !== null) {
      m.shared.pending = null;
      var F = M, j = F.next;
      F.next = null, C === null ? A = j : C.next = j, C = F;
      var Y = i.alternate;
      Y !== null && (Y = Y.updateQueue, M = Y.lastBaseUpdate, M !== C && (M === null ? Y.firstBaseUpdate = j : M.next = j, Y.lastBaseUpdate = F));
    }
    if (A !== null) {
      var J = m.baseState;
      C = 0, Y = j = F = null, M = A;
      do {
        var G = M.lane, ut = M.eventTime;
        if ((d & G) === G) {
          Y !== null && (Y = Y.next = {
            eventTime: ut,
            lane: 0,
            tag: M.tag,
            payload: M.payload,
            callback: M.callback,
            next: null
          });
          t: {
            var mt = i, yt = M;
            switch (G = r, ut = c, yt.tag) {
              case 1:
                if (mt = yt.payload, typeof mt == "function") {
                  J = mt.call(ut, J, G);
                  break t;
                }
                J = mt;
                break t;
              case 3:
                mt.flags = mt.flags & -65537 | 128;
              case 0:
                if (mt = yt.payload, G = typeof mt == "function" ? mt.call(ut, J, G) : mt, G == null) break t;
                J = st({}, J, G);
                break t;
              case 2:
                Eo = !0;
            }
          }
          M.callback !== null && M.lane !== 0 && (i.flags |= 64, G = m.effects, G === null ? m.effects = [M] : G.push(M));
        } else ut = { eventTime: ut, lane: G, tag: M.tag, payload: M.payload, callback: M.callback, next: null }, Y === null ? (j = Y = ut, F = J) : Y = Y.next = ut, C |= G;
        if (M = M.next, M === null) {
          if (M = m.shared.pending, M === null) break;
          G = M, M = G.next, G.next = null, m.lastBaseUpdate = G, m.shared.pending = null;
        }
      } while (!0);
      if (Y === null && (F = J), m.baseState = F, m.firstBaseUpdate = j, m.lastBaseUpdate = Y, r = m.shared.interleaved, r !== null) {
        m = r;
        do
          C |= m.lane, m = m.next;
        while (m !== r);
      } else A === null && (m.shared.lanes = 0);
      Ha |= C, i.lanes = C, i.memoizedState = J;
    }
  }
  function QS(i, r, c) {
    if (i = r.effects, r.effects = null, i !== null) for (r = 0; r < i.length; r++) {
      var d = i[r], m = d.callback;
      if (m !== null) {
        if (d.callback = null, d = c, typeof m != "function") throw Error(e(191, m));
        m.call(d);
      }
    }
  }
  var rd = {}, Ds = Ao(rd), od = Ao(rd), ad = Ao(rd);
  function za(i) {
    if (i === rd) throw Error(e(174));
    return i;
  }
  function kv(i, r) {
    switch (ue(ad, r), ue(od, i), ue(Ds, rd), i = r.nodeType, i) {
      case 9:
      case 11:
        r = (r = r.documentElement) ? r.namespaceURI : Ii(null, "");
        break;
      default:
        i = i === 8 ? r.parentNode : r, r = i.namespaceURI || null, i = i.tagName, r = Ii(r, i);
    }
    ge(Ds), ue(Ds, r);
  }
  function Dc() {
    ge(Ds), ge(od), ge(ad);
  }
  function ZS(i) {
    za(ad.current);
    var r = za(Ds.current), c = Ii(r, i.type);
    r !== c && (ue(od, i), ue(Ds, c));
  }
  function Pv(i) {
    od.current === i && (ge(Ds), ge(od));
  }
  var Se = Ao(0);
  function xg(i) {
    for (var r = i; r !== null; ) {
      if (r.tag === 13) {
        var c = r.memoizedState;
        if (c !== null && (c = c.dehydrated, c === null || c.data === "$?" || c.data === "$!")) return r;
      } else if (r.tag === 19 && r.memoizedProps.revealOrder !== void 0) {
        if ((r.flags & 128) !== 0) return r;
      } else if (r.child !== null) {
        r.child.return = r, r = r.child;
        continue;
      }
      if (r === i) break;
      for (; r.sibling === null; ) {
        if (r.return === null || r.return === i) return null;
        r = r.return;
      }
      r.sibling.return = r.return, r = r.sibling;
    }
    return null;
  }
  var Rv = [];
  function Mv() {
    for (var i = 0; i < Rv.length; i++) Rv[i]._workInProgressVersionPrimary = null;
    Rv.length = 0;
  }
  var Tg = L.ReactCurrentDispatcher, Lv = L.ReactCurrentBatchConfig, Ua = 0, be = null, Ve = null, Ze = null, kg = !1, ld = !1, cd = 0, oT = 0;
  function bn() {
    throw Error(e(321));
  }
  function Dv(i, r) {
    if (r === null) return !1;
    for (var c = 0; c < r.length && c < i.length; c++) if (!as(i[c], r[c])) return !1;
    return !0;
  }
  function Iv(i, r, c, d, m, A) {
    if (Ua = A, be = r, r.memoizedState = null, r.updateQueue = null, r.lanes = 0, Tg.current = i === null || i.memoizedState === null ? uT : hT, i = c(d, m), ld) {
      A = 0;
      do {
        if (ld = !1, cd = 0, 25 <= A) throw Error(e(301));
        A += 1, Ze = Ve = null, r.updateQueue = null, Tg.current = dT, i = c(d, m);
      } while (ld);
    }
    if (Tg.current = Mg, r = Ve !== null && Ve.next !== null, Ua = 0, Ze = Ve = be = null, kg = !1, r) throw Error(e(300));
    return i;
  }
  function Fv() {
    var i = cd !== 0;
    return cd = 0, i;
  }
  function Is() {
    var i = { memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null };
    return Ze === null ? be.memoizedState = Ze = i : Ze = Ze.next = i, Ze;
  }
  function Bi() {
    if (Ve === null) {
      var i = be.alternate;
      i = i !== null ? i.memoizedState : null;
    } else i = Ve.next;
    var r = Ze === null ? be.memoizedState : Ze.next;
    if (r !== null) Ze = r, Ve = i;
    else {
      if (i === null) throw Error(e(310));
      Ve = i, i = { memoizedState: Ve.memoizedState, baseState: Ve.baseState, baseQueue: Ve.baseQueue, queue: Ve.queue, next: null }, Ze === null ? be.memoizedState = Ze = i : Ze = Ze.next = i;
    }
    return Ze;
  }
  function ud(i, r) {
    return typeof r == "function" ? r(i) : r;
  }
  function Nv(i) {
    var r = Bi(), c = r.queue;
    if (c === null) throw Error(e(311));
    c.lastRenderedReducer = i;
    var d = Ve, m = d.baseQueue, A = c.pending;
    if (A !== null) {
      if (m !== null) {
        var C = m.next;
        m.next = A.next, A.next = C;
      }
      d.baseQueue = m = A, c.pending = null;
    }
    if (m !== null) {
      A = m.next, d = d.baseState;
      var M = C = null, F = null, j = A;
      do {
        var Y = j.lane;
        if ((Ua & Y) === Y) F !== null && (F = F.next = { lane: 0, action: j.action, hasEagerState: j.hasEagerState, eagerState: j.eagerState, next: null }), d = j.hasEagerState ? j.eagerState : i(d, j.action);
        else {
          var J = {
            lane: Y,
            action: j.action,
            hasEagerState: j.hasEagerState,
            eagerState: j.eagerState,
            next: null
          };
          F === null ? (M = F = J, C = d) : F = F.next = J, be.lanes |= Y, Ha |= Y;
        }
        j = j.next;
      } while (j !== null && j !== A);
      F === null ? C = d : F.next = M, as(d, r.memoizedState) || (Gn = !0), r.memoizedState = d, r.baseState = C, r.baseQueue = F, c.lastRenderedState = d;
    }
    if (i = c.interleaved, i !== null) {
      m = i;
      do
        A = m.lane, be.lanes |= A, Ha |= A, m = m.next;
      while (m !== i);
    } else m === null && (c.lanes = 0);
    return [r.memoizedState, c.dispatch];
  }
  function Ov(i) {
    var r = Bi(), c = r.queue;
    if (c === null) throw Error(e(311));
    c.lastRenderedReducer = i;
    var d = c.dispatch, m = c.pending, A = r.memoizedState;
    if (m !== null) {
      c.pending = null;
      var C = m = m.next;
      do
        A = i(A, C.action), C = C.next;
      while (C !== m);
      as(A, r.memoizedState) || (Gn = !0), r.memoizedState = A, r.baseQueue === null && (r.baseState = A), c.lastRenderedState = A;
    }
    return [A, d];
  }
  function JS() {
  }
  function tb(i, r) {
    var c = be, d = Bi(), m = r(), A = !as(d.memoizedState, m);
    if (A && (d.memoizedState = m, Gn = !0), d = d.queue, Bv(ib.bind(null, c, d, i), [i]), d.getSnapshot !== r || A || Ze !== null && Ze.memoizedState.tag & 1) {
      if (c.flags |= 2048, hd(9, nb.bind(null, c, d, m, r), void 0, null), Je === null) throw Error(e(349));
      (Ua & 30) !== 0 || eb(c, r, m);
    }
    return m;
  }
  function eb(i, r, c) {
    i.flags |= 16384, i = { getSnapshot: r, value: c }, r = be.updateQueue, r === null ? (r = { lastEffect: null, stores: null }, be.updateQueue = r, r.stores = [i]) : (c = r.stores, c === null ? r.stores = [i] : c.push(i));
  }
  function nb(i, r, c, d) {
    r.value = c, r.getSnapshot = d, sb(r) && rb(i);
  }
  function ib(i, r, c) {
    return c(function() {
      sb(r) && rb(i);
    });
  }
  function sb(i) {
    var r = i.getSnapshot;
    i = i.value;
    try {
      var c = r();
      return !as(i, c);
    } catch {
      return !0;
    }
  }
  function rb(i) {
    var r = mr(i, 1);
    r !== null && ds(r, i, 1, -1);
  }
  function ob(i) {
    var r = Is();
    return typeof i == "function" && (i = i()), r.memoizedState = r.baseState = i, i = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: ud, lastRenderedState: i }, r.queue = i, i = i.dispatch = cT.bind(null, be, i), [r.memoizedState, i];
  }
  function hd(i, r, c, d) {
    return i = { tag: i, create: r, destroy: c, deps: d, next: null }, r = be.updateQueue, r === null ? (r = { lastEffect: null, stores: null }, be.updateQueue = r, r.lastEffect = i.next = i) : (c = r.lastEffect, c === null ? r.lastEffect = i.next = i : (d = c.next, c.next = i, i.next = d, r.lastEffect = i)), i;
  }
  function ab() {
    return Bi().memoizedState;
  }
  function Pg(i, r, c, d) {
    var m = Is();
    be.flags |= i, m.memoizedState = hd(1 | r, c, void 0, d === void 0 ? null : d);
  }
  function Rg(i, r, c, d) {
    var m = Bi();
    d = d === void 0 ? null : d;
    var A = void 0;
    if (Ve !== null) {
      var C = Ve.memoizedState;
      if (A = C.destroy, d !== null && Dv(d, C.deps)) {
        m.memoizedState = hd(r, c, A, d);
        return;
      }
    }
    be.flags |= i, m.memoizedState = hd(1 | r, c, A, d);
  }
  function lb(i, r) {
    return Pg(8390656, 8, i, r);
  }
  function Bv(i, r) {
    return Rg(2048, 8, i, r);
  }
  function cb(i, r) {
    return Rg(4, 2, i, r);
  }
  function ub(i, r) {
    return Rg(4, 4, i, r);
  }
  function hb(i, r) {
    if (typeof r == "function") return i = i(), r(i), function() {
      r(null);
    };
    if (r != null) return i = i(), r.current = i, function() {
      r.current = null;
    };
  }
  function db(i, r, c) {
    return c = c != null ? c.concat([i]) : null, Rg(4, 4, hb.bind(null, r, i), c);
  }
  function zv() {
  }
  function fb(i, r) {
    var c = Bi();
    r = r === void 0 ? null : r;
    var d = c.memoizedState;
    return d !== null && r !== null && Dv(r, d[1]) ? d[0] : (c.memoizedState = [i, r], i);
  }
  function pb(i, r) {
    var c = Bi();
    r = r === void 0 ? null : r;
    var d = c.memoizedState;
    return d !== null && r !== null && Dv(r, d[1]) ? d[0] : (i = i(), c.memoizedState = [i, r], i);
  }
  function gb(i, r, c) {
    return (Ua & 21) === 0 ? (i.baseState && (i.baseState = !1, Gn = !0), i.memoizedState = c) : (as(c, r) || (c = WA(), be.lanes |= c, Ha |= c, i.baseState = !0), r);
  }
  function aT(i, r) {
    var c = Kt;
    Kt = c !== 0 && 4 > c ? c : 4, i(!0);
    var d = Lv.transition;
    Lv.transition = {};
    try {
      i(!1), r();
    } finally {
      Kt = c, Lv.transition = d;
    }
  }
  function mb() {
    return Bi().memoizedState;
  }
  function lT(i, r, c) {
    var d = ko(i);
    if (c = { lane: d, action: c, hasEagerState: !1, eagerState: null, next: null }, yb(i)) vb(r, c);
    else if (c = qS(i, r, c, d), c !== null) {
      var m = Fn();
      ds(c, i, d, m), wb(c, r, d);
    }
  }
  function cT(i, r, c) {
    var d = ko(i), m = { lane: d, action: c, hasEagerState: !1, eagerState: null, next: null };
    if (yb(i)) vb(r, m);
    else {
      var A = i.alternate;
      if (i.lanes === 0 && (A === null || A.lanes === 0) && (A = r.lastRenderedReducer, A !== null)) try {
        var C = r.lastRenderedState, M = A(C, c);
        if (m.hasEagerState = !0, m.eagerState = M, as(M, C)) {
          var F = r.interleaved;
          F === null ? (m.next = m, xv(r)) : (m.next = F.next, F.next = m), r.interleaved = m;
          return;
        }
      } catch {
      } finally {
      }
      c = qS(i, r, m, d), c !== null && (m = Fn(), ds(c, i, d, m), wb(c, r, d));
    }
  }
  function yb(i) {
    var r = i.alternate;
    return i === be || r !== null && r === be;
  }
  function vb(i, r) {
    ld = kg = !0;
    var c = i.pending;
    c === null ? r.next = r : (r.next = c.next, c.next = r), i.pending = r;
  }
  function wb(i, r, c) {
    if ((c & 4194240) !== 0) {
      var d = r.lanes;
      d &= i.pendingLanes, c |= d, r.lanes = c, Hy(i, c);
    }
  }
  var Mg = { readContext: Oi, useCallback: bn, useContext: bn, useEffect: bn, useImperativeHandle: bn, useInsertionEffect: bn, useLayoutEffect: bn, useMemo: bn, useReducer: bn, useRef: bn, useState: bn, useDebugValue: bn, useDeferredValue: bn, useTransition: bn, useMutableSource: bn, useSyncExternalStore: bn, useId: bn, unstable_isNewReconciler: !1 }, uT = { readContext: Oi, useCallback: function(i, r) {
    return Is().memoizedState = [i, r === void 0 ? null : r], i;
  }, useContext: Oi, useEffect: lb, useImperativeHandle: function(i, r, c) {
    return c = c != null ? c.concat([i]) : null, Pg(
      4194308,
      4,
      hb.bind(null, r, i),
      c
    );
  }, useLayoutEffect: function(i, r) {
    return Pg(4194308, 4, i, r);
  }, useInsertionEffect: function(i, r) {
    return Pg(4, 2, i, r);
  }, useMemo: function(i, r) {
    var c = Is();
    return r = r === void 0 ? null : r, i = i(), c.memoizedState = [i, r], i;
  }, useReducer: function(i, r, c) {
    var d = Is();
    return r = c !== void 0 ? c(r) : r, d.memoizedState = d.baseState = r, i = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: i, lastRenderedState: r }, d.queue = i, i = i.dispatch = lT.bind(null, be, i), [d.memoizedState, i];
  }, useRef: function(i) {
    var r = Is();
    return i = { current: i }, r.memoizedState = i;
  }, useState: ob, useDebugValue: zv, useDeferredValue: function(i) {
    return Is().memoizedState = i;
  }, useTransition: function() {
    var i = ob(!1), r = i[0];
    return i = aT.bind(null, i[1]), Is().memoizedState = i, [r, i];
  }, useMutableSource: function() {
  }, useSyncExternalStore: function(i, r, c) {
    var d = be, m = Is();
    if (ye) {
      if (c === void 0) throw Error(e(407));
      c = c();
    } else {
      if (c = r(), Je === null) throw Error(e(349));
      (Ua & 30) !== 0 || eb(d, r, c);
    }
    m.memoizedState = c;
    var A = { value: c, getSnapshot: r };
    return m.queue = A, lb(ib.bind(
      null,
      d,
      A,
      i
    ), [i]), d.flags |= 2048, hd(9, nb.bind(null, d, A, c, r), void 0, null), c;
  }, useId: function() {
    var i = Is(), r = Je.identifierPrefix;
    if (ye) {
      var c = gr, d = pr;
      c = (d & ~(1 << 32 - os(d) - 1)).toString(32) + c, r = ":" + r + "R" + c, c = cd++, 0 < c && (r += "H" + c.toString(32)), r += ":";
    } else c = oT++, r = ":" + r + "r" + c.toString(32) + ":";
    return i.memoizedState = r;
  }, unstable_isNewReconciler: !1 }, hT = {
    readContext: Oi,
    useCallback: fb,
    useContext: Oi,
    useEffect: Bv,
    useImperativeHandle: db,
    useInsertionEffect: cb,
    useLayoutEffect: ub,
    useMemo: pb,
    useReducer: Nv,
    useRef: ab,
    useState: function() {
      return Nv(ud);
    },
    useDebugValue: zv,
    useDeferredValue: function(i) {
      var r = Bi();
      return gb(r, Ve.memoizedState, i);
    },
    useTransition: function() {
      var i = Nv(ud)[0], r = Bi().memoizedState;
      return [i, r];
    },
    useMutableSource: JS,
    useSyncExternalStore: tb,
    useId: mb,
    unstable_isNewReconciler: !1
  }, dT = { readContext: Oi, useCallback: fb, useContext: Oi, useEffect: Bv, useImperativeHandle: db, useInsertionEffect: cb, useLayoutEffect: ub, useMemo: pb, useReducer: Ov, useRef: ab, useState: function() {
    return Ov(ud);
  }, useDebugValue: zv, useDeferredValue: function(i) {
    var r = Bi();
    return Ve === null ? r.memoizedState = i : gb(r, Ve.memoizedState, i);
  }, useTransition: function() {
    var i = Ov(ud)[0], r = Bi().memoizedState;
    return [i, r];
  }, useMutableSource: JS, useSyncExternalStore: tb, useId: mb, unstable_isNewReconciler: !1 };
  function cs(i, r) {
    if (i && i.defaultProps) {
      r = st({}, r), i = i.defaultProps;
      for (var c in i) r[c] === void 0 && (r[c] = i[c]);
      return r;
    }
    return r;
  }
  function Uv(i, r, c, d) {
    r = i.memoizedState, c = c(d, r), c = c == null ? r : st({}, r, c), i.memoizedState = c, i.lanes === 0 && (i.updateQueue.baseState = c);
  }
  var Lg = { isMounted: function(i) {
    return (i = i._reactInternals) ? Da(i) === i : !1;
  }, enqueueSetState: function(i, r, c) {
    i = i._reactInternals;
    var d = Fn(), m = ko(i), A = yr(d, m);
    A.payload = r, c != null && (A.callback = c), r = _o(i, A, m), r !== null && (ds(r, i, m, d), _g(r, i, m));
  }, enqueueReplaceState: function(i, r, c) {
    i = i._reactInternals;
    var d = Fn(), m = ko(i), A = yr(d, m);
    A.tag = 1, A.payload = r, c != null && (A.callback = c), r = _o(i, A, m), r !== null && (ds(r, i, m, d), _g(r, i, m));
  }, enqueueForceUpdate: function(i, r) {
    i = i._reactInternals;
    var c = Fn(), d = ko(i), m = yr(c, d);
    m.tag = 2, r != null && (m.callback = r), r = _o(i, m, d), r !== null && (ds(r, i, d, c), _g(r, i, d));
  } };
  function Ab(i, r, c, d, m, A, C) {
    return i = i.stateNode, typeof i.shouldComponentUpdate == "function" ? i.shouldComponentUpdate(d, A, C) : r.prototype && r.prototype.isPureReactComponent ? !Qh(c, d) || !Qh(m, A) : !0;
  }
  function Sb(i, r, c) {
    var d = !1, m = So, A = r.contextType;
    return typeof A == "object" && A !== null ? A = Oi(A) : (m = Wn(r) ? Fa : Sn.current, d = r.contextTypes, A = (d = d != null) ? xc(i, m) : So), r = new r(c, A), i.memoizedState = r.state !== null && r.state !== void 0 ? r.state : null, r.updater = Lg, i.stateNode = r, r._reactInternals = i, d && (i = i.stateNode, i.__reactInternalMemoizedUnmaskedChildContext = m, i.__reactInternalMemoizedMaskedChildContext = A), r;
  }
  function bb(i, r, c, d) {
    i = r.state, typeof r.componentWillReceiveProps == "function" && r.componentWillReceiveProps(c, d), typeof r.UNSAFE_componentWillReceiveProps == "function" && r.UNSAFE_componentWillReceiveProps(c, d), r.state !== i && Lg.enqueueReplaceState(r, r.state, null);
  }
  function Hv(i, r, c, d) {
    var m = i.stateNode;
    m.props = c, m.state = i.memoizedState, m.refs = {}, Tv(i);
    var A = r.contextType;
    typeof A == "object" && A !== null ? m.context = Oi(A) : (A = Wn(r) ? Fa : Sn.current, m.context = xc(i, A)), m.state = i.memoizedState, A = r.getDerivedStateFromProps, typeof A == "function" && (Uv(i, r, A, c), m.state = i.memoizedState), typeof r.getDerivedStateFromProps == "function" || typeof m.getSnapshotBeforeUpdate == "function" || typeof m.UNSAFE_componentWillMount != "function" && typeof m.componentWillMount != "function" || (r = m.state, typeof m.componentWillMount == "function" && m.componentWillMount(), typeof m.UNSAFE_componentWillMount == "function" && m.UNSAFE_componentWillMount(), r !== m.state && Lg.enqueueReplaceState(m, m.state, null), Cg(i, c, m, d), m.state = i.memoizedState), typeof m.componentDidMount == "function" && (i.flags |= 4194308);
  }
  function Ic(i, r) {
    try {
      var c = "", d = r;
      do
        c += Et(d), d = d.return;
      while (d);
      var m = c;
    } catch (A) {
      m = `
Error generating stack: ` + A.message + `
` + A.stack;
    }
    return { value: i, source: r, stack: m, digest: null };
  }
  function jv(i, r, c) {
    return { value: i, source: null, stack: c ?? null, digest: r ?? null };
  }
  function $v(i, r) {
    try {
      console.error(r.value);
    } catch (c) {
      setTimeout(function() {
        throw c;
      });
    }
  }
  var fT = typeof WeakMap == "function" ? WeakMap : Map;
  function Eb(i, r, c) {
    c = yr(-1, c), c.tag = 3, c.payload = { element: null };
    var d = r.value;
    return c.callback = function() {
      zg || (zg = !0, s0 = d), $v(i, r);
    }, c;
  }
  function _b(i, r, c) {
    c = yr(-1, c), c.tag = 3;
    var d = i.type.getDerivedStateFromError;
    if (typeof d == "function") {
      var m = r.value;
      c.payload = function() {
        return d(m);
      }, c.callback = function() {
        $v(i, r);
      };
    }
    var A = i.stateNode;
    return A !== null && typeof A.componentDidCatch == "function" && (c.callback = function() {
      $v(i, r), typeof d != "function" && (xo === null ? xo = /* @__PURE__ */ new Set([this]) : xo.add(this));
      var C = r.stack;
      this.componentDidCatch(r.value, { componentStack: C !== null ? C : "" });
    }), c;
  }
  function Cb(i, r, c) {
    var d = i.pingCache;
    if (d === null) {
      d = i.pingCache = new fT();
      var m = /* @__PURE__ */ new Set();
      d.set(r, m);
    } else m = d.get(r), m === void 0 && (m = /* @__PURE__ */ new Set(), d.set(r, m));
    m.has(c) || (m.add(c), i = TT.bind(null, i, r, c), r.then(i, i));
  }
  function xb(i) {
    do {
      var r;
      if ((r = i.tag === 13) && (r = i.memoizedState, r = r !== null ? r.dehydrated !== null : !0), r) return i;
      i = i.return;
    } while (i !== null);
    return null;
  }
  function Tb(i, r, c, d, m) {
    return (i.mode & 1) === 0 ? (i === r ? i.flags |= 65536 : (i.flags |= 128, c.flags |= 131072, c.flags &= -52805, c.tag === 1 && (c.alternate === null ? c.tag = 17 : (r = yr(-1, 1), r.tag = 2, _o(c, r, 1))), c.lanes |= 1), i) : (i.flags |= 65536, i.lanes = m, i);
  }
  var pT = L.ReactCurrentOwner, Gn = !1;
  function In(i, r, c, d) {
    r.child = i === null ? XS(r, null, c, d) : Rc(r, i.child, c, d);
  }
  function kb(i, r, c, d, m) {
    c = c.render;
    var A = r.ref;
    return Lc(r, m), d = Iv(i, r, c, d, A, m), c = Fv(), i !== null && !Gn ? (r.updateQueue = i.updateQueue, r.flags &= -2053, i.lanes &= ~m, vr(i, r, m)) : (ye && c && yv(r), r.flags |= 1, In(i, r, d, m), r.child);
  }
  function Pb(i, r, c, d, m) {
    if (i === null) {
      var A = c.type;
      return typeof A == "function" && !h0(A) && A.defaultProps === void 0 && c.compare === null && c.defaultProps === void 0 ? (r.tag = 15, r.type = A, Rb(i, r, A, d, m)) : (i = Wg(c.type, null, d, r, r.mode, m), i.ref = r.ref, i.return = r, r.child = i);
    }
    if (A = i.child, (i.lanes & m) === 0) {
      var C = A.memoizedProps;
      if (c = c.compare, c = c !== null ? c : Qh, c(C, d) && i.ref === r.ref) return vr(i, r, m);
    }
    return r.flags |= 1, i = Ro(A, d), i.ref = r.ref, i.return = r, r.child = i;
  }
  function Rb(i, r, c, d, m) {
    if (i !== null) {
      var A = i.memoizedProps;
      if (Qh(A, d) && i.ref === r.ref) if (Gn = !1, r.pendingProps = d = A, (i.lanes & m) !== 0) (i.flags & 131072) !== 0 && (Gn = !0);
      else return r.lanes = i.lanes, vr(i, r, m);
    }
    return Vv(i, r, c, d, m);
  }
  function Mb(i, r, c) {
    var d = r.pendingProps, m = d.children, A = i !== null ? i.memoizedState : null;
    if (d.mode === "hidden") if ((r.mode & 1) === 0) r.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, ue(Nc, ci), ci |= c;
    else {
      if ((c & 1073741824) === 0) return i = A !== null ? A.baseLanes | c : c, r.lanes = r.childLanes = 1073741824, r.memoizedState = { baseLanes: i, cachePool: null, transitions: null }, r.updateQueue = null, ue(Nc, ci), ci |= i, null;
      r.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, d = A !== null ? A.baseLanes : c, ue(Nc, ci), ci |= d;
    }
    else A !== null ? (d = A.baseLanes | c, r.memoizedState = null) : d = c, ue(Nc, ci), ci |= d;
    return In(i, r, m, c), r.child;
  }
  function Lb(i, r) {
    var c = r.ref;
    (i === null && c !== null || i !== null && i.ref !== c) && (r.flags |= 512, r.flags |= 2097152);
  }
  function Vv(i, r, c, d, m) {
    var A = Wn(c) ? Fa : Sn.current;
    return A = xc(r, A), Lc(r, m), c = Iv(i, r, c, d, A, m), d = Fv(), i !== null && !Gn ? (r.updateQueue = i.updateQueue, r.flags &= -2053, i.lanes &= ~m, vr(i, r, m)) : (ye && d && yv(r), r.flags |= 1, In(i, r, c, m), r.child);
  }
  function Db(i, r, c, d, m) {
    if (Wn(c)) {
      var A = !0;
      mg(r);
    } else A = !1;
    if (Lc(r, m), r.stateNode === null) Ig(i, r), Sb(r, c, d), Hv(r, c, d, m), d = !0;
    else if (i === null) {
      var C = r.stateNode, M = r.memoizedProps;
      C.props = M;
      var F = C.context, j = c.contextType;
      typeof j == "object" && j !== null ? j = Oi(j) : (j = Wn(c) ? Fa : Sn.current, j = xc(r, j));
      var Y = c.getDerivedStateFromProps, J = typeof Y == "function" || typeof C.getSnapshotBeforeUpdate == "function";
      J || typeof C.UNSAFE_componentWillReceiveProps != "function" && typeof C.componentWillReceiveProps != "function" || (M !== d || F !== j) && bb(r, C, d, j), Eo = !1;
      var G = r.memoizedState;
      C.state = G, Cg(r, d, C, m), F = r.memoizedState, M !== d || G !== F || Vn.current || Eo ? (typeof Y == "function" && (Uv(r, c, Y, d), F = r.memoizedState), (M = Eo || Ab(r, c, M, d, G, F, j)) ? (J || typeof C.UNSAFE_componentWillMount != "function" && typeof C.componentWillMount != "function" || (typeof C.componentWillMount == "function" && C.componentWillMount(), typeof C.UNSAFE_componentWillMount == "function" && C.UNSAFE_componentWillMount()), typeof C.componentDidMount == "function" && (r.flags |= 4194308)) : (typeof C.componentDidMount == "function" && (r.flags |= 4194308), r.memoizedProps = d, r.memoizedState = F), C.props = d, C.state = F, C.context = j, d = M) : (typeof C.componentDidMount == "function" && (r.flags |= 4194308), d = !1);
    } else {
      C = r.stateNode, YS(i, r), M = r.memoizedProps, j = r.type === r.elementType ? M : cs(r.type, M), C.props = j, J = r.pendingProps, G = C.context, F = c.contextType, typeof F == "object" && F !== null ? F = Oi(F) : (F = Wn(c) ? Fa : Sn.current, F = xc(r, F));
      var ut = c.getDerivedStateFromProps;
      (Y = typeof ut == "function" || typeof C.getSnapshotBeforeUpdate == "function") || typeof C.UNSAFE_componentWillReceiveProps != "function" && typeof C.componentWillReceiveProps != "function" || (M !== J || G !== F) && bb(r, C, d, F), Eo = !1, G = r.memoizedState, C.state = G, Cg(r, d, C, m);
      var mt = r.memoizedState;
      M !== J || G !== mt || Vn.current || Eo ? (typeof ut == "function" && (Uv(r, c, ut, d), mt = r.memoizedState), (j = Eo || Ab(r, c, j, d, G, mt, F) || !1) ? (Y || typeof C.UNSAFE_componentWillUpdate != "function" && typeof C.componentWillUpdate != "function" || (typeof C.componentWillUpdate == "function" && C.componentWillUpdate(d, mt, F), typeof C.UNSAFE_componentWillUpdate == "function" && C.UNSAFE_componentWillUpdate(d, mt, F)), typeof C.componentDidUpdate == "function" && (r.flags |= 4), typeof C.getSnapshotBeforeUpdate == "function" && (r.flags |= 1024)) : (typeof C.componentDidUpdate != "function" || M === i.memoizedProps && G === i.memoizedState || (r.flags |= 4), typeof C.getSnapshotBeforeUpdate != "function" || M === i.memoizedProps && G === i.memoizedState || (r.flags |= 1024), r.memoizedProps = d, r.memoizedState = mt), C.props = d, C.state = mt, C.context = F, d = j) : (typeof C.componentDidUpdate != "function" || M === i.memoizedProps && G === i.memoizedState || (r.flags |= 4), typeof C.getSnapshotBeforeUpdate != "function" || M === i.memoizedProps && G === i.memoizedState || (r.flags |= 1024), d = !1);
    }
    return Wv(i, r, c, d, A, m);
  }
  function Wv(i, r, c, d, m, A) {
    Lb(i, r);
    var C = (r.flags & 128) !== 0;
    if (!d && !C) return m && BS(r, c, !1), vr(i, r, A);
    d = r.stateNode, pT.current = r;
    var M = C && typeof c.getDerivedStateFromError != "function" ? null : d.render();
    return r.flags |= 1, i !== null && C ? (r.child = Rc(r, i.child, null, A), r.child = Rc(r, null, M, A)) : In(i, r, M, A), r.memoizedState = d.state, m && BS(r, c, !0), r.child;
  }
  function Ib(i) {
    var r = i.stateNode;
    r.pendingContext ? NS(i, r.pendingContext, r.pendingContext !== r.context) : r.context && NS(i, r.context, !1), kv(i, r.containerInfo);
  }
  function Fb(i, r, c, d, m) {
    return Pc(), Sv(m), r.flags |= 256, In(i, r, c, d), r.child;
  }
  var Gv = { dehydrated: null, treeContext: null, retryLane: 0 };
  function Xv(i) {
    return { baseLanes: i, cachePool: null, transitions: null };
  }
  function Nb(i, r, c) {
    var d = r.pendingProps, m = Se.current, A = !1, C = (r.flags & 128) !== 0, M;
    if ((M = C) || (M = i !== null && i.memoizedState === null ? !1 : (m & 2) !== 0), M ? (A = !0, r.flags &= -129) : (i === null || i.memoizedState !== null) && (m |= 1), ue(Se, m & 1), i === null)
      return Av(r), i = r.memoizedState, i !== null && (i = i.dehydrated, i !== null) ? ((r.mode & 1) === 0 ? r.lanes = 1 : i.data === "$!" ? r.lanes = 8 : r.lanes = 1073741824, null) : (C = d.children, i = d.fallback, A ? (d = r.mode, A = r.child, C = { mode: "hidden", children: C }, (d & 1) === 0 && A !== null ? (A.childLanes = 0, A.pendingProps = C) : A = Gg(C, d, 0, null), i = Wa(i, d, c, null), A.return = r, i.return = r, A.sibling = i, r.child = A, r.child.memoizedState = Xv(c), r.memoizedState = Gv, i) : qv(r, C));
    if (m = i.memoizedState, m !== null && (M = m.dehydrated, M !== null)) return gT(i, r, C, d, M, m, c);
    if (A) {
      A = d.fallback, C = r.mode, m = i.child, M = m.sibling;
      var F = { mode: "hidden", children: d.children };
      return (C & 1) === 0 && r.child !== m ? (d = r.child, d.childLanes = 0, d.pendingProps = F, r.deletions = null) : (d = Ro(m, F), d.subtreeFlags = m.subtreeFlags & 14680064), M !== null ? A = Ro(M, A) : (A = Wa(A, C, c, null), A.flags |= 2), A.return = r, d.return = r, d.sibling = A, r.child = d, d = A, A = r.child, C = i.child.memoizedState, C = C === null ? Xv(c) : { baseLanes: C.baseLanes | c, cachePool: null, transitions: C.transitions }, A.memoizedState = C, A.childLanes = i.childLanes & ~c, r.memoizedState = Gv, d;
    }
    return A = i.child, i = A.sibling, d = Ro(A, { mode: "visible", children: d.children }), (r.mode & 1) === 0 && (d.lanes = c), d.return = r, d.sibling = null, i !== null && (c = r.deletions, c === null ? (r.deletions = [i], r.flags |= 16) : c.push(i)), r.child = d, r.memoizedState = null, d;
  }
  function qv(i, r) {
    return r = Gg({ mode: "visible", children: r }, i.mode, 0, null), r.return = i, i.child = r;
  }
  function Dg(i, r, c, d) {
    return d !== null && Sv(d), Rc(r, i.child, null, c), i = qv(r, r.pendingProps.children), i.flags |= 2, r.memoizedState = null, i;
  }
  function gT(i, r, c, d, m, A, C) {
    if (c)
      return r.flags & 256 ? (r.flags &= -257, d = jv(Error(e(422))), Dg(i, r, C, d)) : r.memoizedState !== null ? (r.child = i.child, r.flags |= 128, null) : (A = d.fallback, m = r.mode, d = Gg({ mode: "visible", children: d.children }, m, 0, null), A = Wa(A, m, C, null), A.flags |= 2, d.return = r, A.return = r, d.sibling = A, r.child = d, (r.mode & 1) !== 0 && Rc(r, i.child, null, C), r.child.memoizedState = Xv(C), r.memoizedState = Gv, A);
    if ((r.mode & 1) === 0) return Dg(i, r, C, null);
    if (m.data === "$!") {
      if (d = m.nextSibling && m.nextSibling.dataset, d) var M = d.dgst;
      return d = M, A = Error(e(419)), d = jv(A, d, void 0), Dg(i, r, C, d);
    }
    if (M = (C & i.childLanes) !== 0, Gn || M) {
      if (d = Je, d !== null) {
        switch (C & -C) {
          case 4:
            m = 2;
            break;
          case 16:
            m = 8;
            break;
          case 64:
          case 128:
          case 256:
          case 512:
          case 1024:
          case 2048:
          case 4096:
          case 8192:
          case 16384:
          case 32768:
          case 65536:
          case 131072:
          case 262144:
          case 524288:
          case 1048576:
          case 2097152:
          case 4194304:
          case 8388608:
          case 16777216:
          case 33554432:
          case 67108864:
            m = 32;
            break;
          case 536870912:
            m = 268435456;
            break;
          default:
            m = 0;
        }
        m = (m & (d.suspendedLanes | C)) !== 0 ? 0 : m, m !== 0 && m !== A.retryLane && (A.retryLane = m, mr(i, m), ds(d, i, m, -1));
      }
      return u0(), d = jv(Error(e(421))), Dg(i, r, C, d);
    }
    return m.data === "$?" ? (r.flags |= 128, r.child = i.child, r = kT.bind(null, i), m._reactRetry = r, null) : (i = A.treeContext, li = wo(m.nextSibling), ai = r, ye = !0, ls = null, i !== null && (Fi[Ni++] = pr, Fi[Ni++] = gr, Fi[Ni++] = Na, pr = i.id, gr = i.overflow, Na = r), r = qv(r, d.children), r.flags |= 4096, r);
  }
  function Ob(i, r, c) {
    i.lanes |= r;
    var d = i.alternate;
    d !== null && (d.lanes |= r), Cv(i.return, r, c);
  }
  function Yv(i, r, c, d, m) {
    var A = i.memoizedState;
    A === null ? i.memoizedState = { isBackwards: r, rendering: null, renderingStartTime: 0, last: d, tail: c, tailMode: m } : (A.isBackwards = r, A.rendering = null, A.renderingStartTime = 0, A.last = d, A.tail = c, A.tailMode = m);
  }
  function Bb(i, r, c) {
    var d = r.pendingProps, m = d.revealOrder, A = d.tail;
    if (In(i, r, d.children, c), d = Se.current, (d & 2) !== 0) d = d & 1 | 2, r.flags |= 128;
    else {
      if (i !== null && (i.flags & 128) !== 0) t: for (i = r.child; i !== null; ) {
        if (i.tag === 13) i.memoizedState !== null && Ob(i, c, r);
        else if (i.tag === 19) Ob(i, c, r);
        else if (i.child !== null) {
          i.child.return = i, i = i.child;
          continue;
        }
        if (i === r) break t;
        for (; i.sibling === null; ) {
          if (i.return === null || i.return === r) break t;
          i = i.return;
        }
        i.sibling.return = i.return, i = i.sibling;
      }
      d &= 1;
    }
    if (ue(Se, d), (r.mode & 1) === 0) r.memoizedState = null;
    else switch (m) {
      case "forwards":
        for (c = r.child, m = null; c !== null; ) i = c.alternate, i !== null && xg(i) === null && (m = c), c = c.sibling;
        c = m, c === null ? (m = r.child, r.child = null) : (m = c.sibling, c.sibling = null), Yv(r, !1, m, c, A);
        break;
      case "backwards":
        for (c = null, m = r.child, r.child = null; m !== null; ) {
          if (i = m.alternate, i !== null && xg(i) === null) {
            r.child = m;
            break;
          }
          i = m.sibling, m.sibling = c, c = m, m = i;
        }
        Yv(r, !0, c, null, A);
        break;
      case "together":
        Yv(r, !1, null, null, void 0);
        break;
      default:
        r.memoizedState = null;
    }
    return r.child;
  }
  function Ig(i, r) {
    (r.mode & 1) === 0 && i !== null && (i.alternate = null, r.alternate = null, r.flags |= 2);
  }
  function vr(i, r, c) {
    if (i !== null && (r.dependencies = i.dependencies), Ha |= r.lanes, (c & r.childLanes) === 0) return null;
    if (i !== null && r.child !== i.child) throw Error(e(153));
    if (r.child !== null) {
      for (i = r.child, c = Ro(i, i.pendingProps), r.child = c, c.return = r; i.sibling !== null; ) i = i.sibling, c = c.sibling = Ro(i, i.pendingProps), c.return = r;
      c.sibling = null;
    }
    return r.child;
  }
  function mT(i, r, c) {
    switch (r.tag) {
      case 3:
        Ib(r), Pc();
        break;
      case 5:
        ZS(r);
        break;
      case 1:
        Wn(r.type) && mg(r);
        break;
      case 4:
        kv(r, r.stateNode.containerInfo);
        break;
      case 10:
        var d = r.type._context, m = r.memoizedProps.value;
        ue(bg, d._currentValue), d._currentValue = m;
        break;
      case 13:
        if (d = r.memoizedState, d !== null)
          return d.dehydrated !== null ? (ue(Se, Se.current & 1), r.flags |= 128, null) : (c & r.child.childLanes) !== 0 ? Nb(i, r, c) : (ue(Se, Se.current & 1), i = vr(i, r, c), i !== null ? i.sibling : null);
        ue(Se, Se.current & 1);
        break;
      case 19:
        if (d = (c & r.childLanes) !== 0, (i.flags & 128) !== 0) {
          if (d) return Bb(i, r, c);
          r.flags |= 128;
        }
        if (m = r.memoizedState, m !== null && (m.rendering = null, m.tail = null, m.lastEffect = null), ue(Se, Se.current), d) break;
        return null;
      case 22:
      case 23:
        return r.lanes = 0, Mb(i, r, c);
    }
    return vr(i, r, c);
  }
  var zb, Kv, Ub, Hb;
  zb = function(i, r) {
    for (var c = r.child; c !== null; ) {
      if (c.tag === 5 || c.tag === 6) i.appendChild(c.stateNode);
      else if (c.tag !== 4 && c.child !== null) {
        c.child.return = c, c = c.child;
        continue;
      }
      if (c === r) break;
      for (; c.sibling === null; ) {
        if (c.return === null || c.return === r) return;
        c = c.return;
      }
      c.sibling.return = c.return, c = c.sibling;
    }
  }, Kv = function() {
  }, Ub = function(i, r, c, d) {
    var m = i.memoizedProps;
    if (m !== d) {
      i = r.stateNode, za(Ds.current);
      var A = null;
      switch (c) {
        case "input":
          m = Jt(i, m), d = Jt(i, d), A = [];
          break;
        case "select":
          m = st({}, m, { value: void 0 }), d = st({}, d, { value: void 0 }), A = [];
          break;
        case "textarea":
          m = Ih(i, m), d = Ih(i, d), A = [];
          break;
        default:
          typeof m.onClick != "function" && typeof d.onClick == "function" && (i.onclick = fg);
      }
      Ry(c, d);
      var C;
      c = null;
      for (j in m) if (!d.hasOwnProperty(j) && m.hasOwnProperty(j) && m[j] != null) if (j === "style") {
        var M = m[j];
        for (C in M) M.hasOwnProperty(C) && (c || (c = {}), c[C] = "");
      } else j !== "dangerouslySetInnerHTML" && j !== "children" && j !== "suppressContentEditableWarning" && j !== "suppressHydrationWarning" && j !== "autoFocus" && (s.hasOwnProperty(j) ? A || (A = []) : (A = A || []).push(j, null));
      for (j in d) {
        var F = d[j];
        if (M = m != null ? m[j] : void 0, d.hasOwnProperty(j) && F !== M && (F != null || M != null)) if (j === "style") if (M) {
          for (C in M) !M.hasOwnProperty(C) || F && F.hasOwnProperty(C) || (c || (c = {}), c[C] = "");
          for (C in F) F.hasOwnProperty(C) && M[C] !== F[C] && (c || (c = {}), c[C] = F[C]);
        } else c || (A || (A = []), A.push(
          j,
          c
        )), c = F;
        else j === "dangerouslySetInnerHTML" ? (F = F ? F.__html : void 0, M = M ? M.__html : void 0, F != null && M !== F && (A = A || []).push(j, F)) : j === "children" ? typeof F != "string" && typeof F != "number" || (A = A || []).push(j, "" + F) : j !== "suppressContentEditableWarning" && j !== "suppressHydrationWarning" && (s.hasOwnProperty(j) ? (F != null && j === "onScroll" && pe("scroll", i), A || M === F || (A = [])) : (A = A || []).push(j, F));
      }
      c && (A = A || []).push("style", c);
      var j = A;
      (r.updateQueue = j) && (r.flags |= 4);
    }
  }, Hb = function(i, r, c, d) {
    c !== d && (r.flags |= 4);
  };
  function dd(i, r) {
    if (!ye) switch (i.tailMode) {
      case "hidden":
        r = i.tail;
        for (var c = null; r !== null; ) r.alternate !== null && (c = r), r = r.sibling;
        c === null ? i.tail = null : c.sibling = null;
        break;
      case "collapsed":
        c = i.tail;
        for (var d = null; c !== null; ) c.alternate !== null && (d = c), c = c.sibling;
        d === null ? r || i.tail === null ? i.tail = null : i.tail.sibling = null : d.sibling = null;
    }
  }
  function En(i) {
    var r = i.alternate !== null && i.alternate.child === i.child, c = 0, d = 0;
    if (r) for (var m = i.child; m !== null; ) c |= m.lanes | m.childLanes, d |= m.subtreeFlags & 14680064, d |= m.flags & 14680064, m.return = i, m = m.sibling;
    else for (m = i.child; m !== null; ) c |= m.lanes | m.childLanes, d |= m.subtreeFlags, d |= m.flags, m.return = i, m = m.sibling;
    return i.subtreeFlags |= d, i.childLanes = c, r;
  }
  function yT(i, r, c) {
    var d = r.pendingProps;
    switch (vv(r), r.tag) {
      case 2:
      case 16:
      case 15:
      case 0:
      case 11:
      case 7:
      case 8:
      case 12:
      case 9:
      case 14:
        return En(r), null;
      case 1:
        return Wn(r.type) && gg(), En(r), null;
      case 3:
        return d = r.stateNode, Dc(), ge(Vn), ge(Sn), Mv(), d.pendingContext && (d.context = d.pendingContext, d.pendingContext = null), (i === null || i.child === null) && (Ag(r) ? r.flags |= 4 : i === null || i.memoizedState.isDehydrated && (r.flags & 256) === 0 || (r.flags |= 1024, ls !== null && (a0(ls), ls = null))), Kv(i, r), En(r), null;
      case 5:
        Pv(r);
        var m = za(ad.current);
        if (c = r.type, i !== null && r.stateNode != null) Ub(i, r, c, d, m), i.ref !== r.ref && (r.flags |= 512, r.flags |= 2097152);
        else {
          if (!d) {
            if (r.stateNode === null) throw Error(e(166));
            return En(r), null;
          }
          if (i = za(Ds.current), Ag(r)) {
            d = r.stateNode, c = r.type;
            var A = r.memoizedProps;
            switch (d[Ls] = r, d[nd] = A, i = (r.mode & 1) !== 0, c) {
              case "dialog":
                pe("cancel", d), pe("close", d);
                break;
              case "iframe":
              case "object":
              case "embed":
                pe("load", d);
                break;
              case "video":
              case "audio":
                for (m = 0; m < Jh.length; m++) pe(Jh[m], d);
                break;
              case "source":
                pe("error", d);
                break;
              case "img":
              case "image":
              case "link":
                pe(
                  "error",
                  d
                ), pe("load", d);
                break;
              case "details":
                pe("toggle", d);
                break;
              case "input":
                Te(d, A), pe("invalid", d);
                break;
              case "select":
                d._wrapperState = { wasMultiple: !!A.multiple }, pe("invalid", d);
                break;
              case "textarea":
                jp(d, A), pe("invalid", d);
            }
            Ry(c, A), m = null;
            for (var C in A) if (A.hasOwnProperty(C)) {
              var M = A[C];
              C === "children" ? typeof M == "string" ? d.textContent !== M && (A.suppressHydrationWarning !== !0 && dg(d.textContent, M, i), m = ["children", M]) : typeof M == "number" && d.textContent !== "" + M && (A.suppressHydrationWarning !== !0 && dg(
                d.textContent,
                M,
                i
              ), m = ["children", "" + M]) : s.hasOwnProperty(C) && M != null && C === "onScroll" && pe("scroll", d);
            }
            switch (c) {
              case "input":
                Qe(d), ho(d, A, !0);
                break;
              case "textarea":
                Qe(d), Vp(d);
                break;
              case "select":
              case "option":
                break;
              default:
                typeof A.onClick == "function" && (d.onclick = fg);
            }
            d = m, r.updateQueue = d, d !== null && (r.flags |= 4);
          } else {
            C = m.nodeType === 9 ? m : m.ownerDocument, i === "http://www.w3.org/1999/xhtml" && (i = Wp(c)), i === "http://www.w3.org/1999/xhtml" ? c === "script" ? (i = C.createElement("div"), i.innerHTML = "<script><\/script>", i = i.removeChild(i.firstChild)) : typeof d.is == "string" ? i = C.createElement(c, { is: d.is }) : (i = C.createElement(c), c === "select" && (C = i, d.multiple ? C.multiple = !0 : d.size && (C.size = d.size))) : i = C.createElementNS(i, c), i[Ls] = r, i[nd] = d, zb(i, r, !1, !1), r.stateNode = i;
            t: {
              switch (C = My(c, d), c) {
                case "dialog":
                  pe("cancel", i), pe("close", i), m = d;
                  break;
                case "iframe":
                case "object":
                case "embed":
                  pe("load", i), m = d;
                  break;
                case "video":
                case "audio":
                  for (m = 0; m < Jh.length; m++) pe(Jh[m], i);
                  m = d;
                  break;
                case "source":
                  pe("error", i), m = d;
                  break;
                case "img":
                case "image":
                case "link":
                  pe(
                    "error",
                    i
                  ), pe("load", i), m = d;
                  break;
                case "details":
                  pe("toggle", i), m = d;
                  break;
                case "input":
                  Te(i, d), m = Jt(i, d), pe("invalid", i);
                  break;
                case "option":
                  m = d;
                  break;
                case "select":
                  i._wrapperState = { wasMultiple: !!d.multiple }, m = st({}, d, { value: void 0 }), pe("invalid", i);
                  break;
                case "textarea":
                  jp(i, d), m = Ih(i, d), pe("invalid", i);
                  break;
                default:
                  m = d;
              }
              Ry(c, m), M = m;
              for (A in M) if (M.hasOwnProperty(A)) {
                var F = M[A];
                A === "style" ? RA(i, F) : A === "dangerouslySetInnerHTML" ? (F = F ? F.__html : void 0, F != null && rs(i, F)) : A === "children" ? typeof F == "string" ? (c !== "textarea" || F !== "") && hr(i, F) : typeof F == "number" && hr(i, "" + F) : A !== "suppressContentEditableWarning" && A !== "suppressHydrationWarning" && A !== "autoFocus" && (s.hasOwnProperty(A) ? F != null && A === "onScroll" && pe("scroll", i) : F != null && P(i, A, F, C));
              }
              switch (c) {
                case "input":
                  Qe(i), ho(i, d, !1);
                  break;
                case "textarea":
                  Qe(i), Vp(i);
                  break;
                case "option":
                  d.value != null && i.setAttribute("value", "" + Ot(d.value));
                  break;
                case "select":
                  i.multiple = !!d.multiple, A = d.value, A != null ? is(i, !!d.multiple, A, !1) : d.defaultValue != null && is(
                    i,
                    !!d.multiple,
                    d.defaultValue,
                    !0
                  );
                  break;
                default:
                  typeof m.onClick == "function" && (i.onclick = fg);
              }
              switch (c) {
                case "button":
                case "input":
                case "select":
                case "textarea":
                  d = !!d.autoFocus;
                  break t;
                case "img":
                  d = !0;
                  break t;
                default:
                  d = !1;
              }
            }
            d && (r.flags |= 4);
          }
          r.ref !== null && (r.flags |= 512, r.flags |= 2097152);
        }
        return En(r), null;
      case 6:
        if (i && r.stateNode != null) Hb(i, r, i.memoizedProps, d);
        else {
          if (typeof d != "string" && r.stateNode === null) throw Error(e(166));
          if (c = za(ad.current), za(Ds.current), Ag(r)) {
            if (d = r.stateNode, c = r.memoizedProps, d[Ls] = r, (A = d.nodeValue !== c) && (i = ai, i !== null)) switch (i.tag) {
              case 3:
                dg(d.nodeValue, c, (i.mode & 1) !== 0);
                break;
              case 5:
                i.memoizedProps.suppressHydrationWarning !== !0 && dg(d.nodeValue, c, (i.mode & 1) !== 0);
            }
            A && (r.flags |= 4);
          } else d = (c.nodeType === 9 ? c : c.ownerDocument).createTextNode(d), d[Ls] = r, r.stateNode = d;
        }
        return En(r), null;
      case 13:
        if (ge(Se), d = r.memoizedState, i === null || i.memoizedState !== null && i.memoizedState.dehydrated !== null) {
          if (ye && li !== null && (r.mode & 1) !== 0 && (r.flags & 128) === 0) VS(), Pc(), r.flags |= 98560, A = !1;
          else if (A = Ag(r), d !== null && d.dehydrated !== null) {
            if (i === null) {
              if (!A) throw Error(e(318));
              if (A = r.memoizedState, A = A !== null ? A.dehydrated : null, !A) throw Error(e(317));
              A[Ls] = r;
            } else Pc(), (r.flags & 128) === 0 && (r.memoizedState = null), r.flags |= 4;
            En(r), A = !1;
          } else ls !== null && (a0(ls), ls = null), A = !0;
          if (!A) return r.flags & 65536 ? r : null;
        }
        return (r.flags & 128) !== 0 ? (r.lanes = c, r) : (d = d !== null, d !== (i !== null && i.memoizedState !== null) && d && (r.child.flags |= 8192, (r.mode & 1) !== 0 && (i === null || (Se.current & 1) !== 0 ? We === 0 && (We = 3) : u0())), r.updateQueue !== null && (r.flags |= 4), En(r), null);
      case 4:
        return Dc(), Kv(i, r), i === null && td(r.stateNode.containerInfo), En(r), null;
      case 10:
        return _v(r.type._context), En(r), null;
      case 17:
        return Wn(r.type) && gg(), En(r), null;
      case 19:
        if (ge(Se), A = r.memoizedState, A === null) return En(r), null;
        if (d = (r.flags & 128) !== 0, C = A.rendering, C === null) if (d) dd(A, !1);
        else {
          if (We !== 0 || i !== null && (i.flags & 128) !== 0) for (i = r.child; i !== null; ) {
            if (C = xg(i), C !== null) {
              for (r.flags |= 128, dd(A, !1), d = C.updateQueue, d !== null && (r.updateQueue = d, r.flags |= 4), r.subtreeFlags = 0, d = c, c = r.child; c !== null; ) A = c, i = d, A.flags &= 14680066, C = A.alternate, C === null ? (A.childLanes = 0, A.lanes = i, A.child = null, A.subtreeFlags = 0, A.memoizedProps = null, A.memoizedState = null, A.updateQueue = null, A.dependencies = null, A.stateNode = null) : (A.childLanes = C.childLanes, A.lanes = C.lanes, A.child = C.child, A.subtreeFlags = 0, A.deletions = null, A.memoizedProps = C.memoizedProps, A.memoizedState = C.memoizedState, A.updateQueue = C.updateQueue, A.type = C.type, i = C.dependencies, A.dependencies = i === null ? null : { lanes: i.lanes, firstContext: i.firstContext }), c = c.sibling;
              return ue(Se, Se.current & 1 | 2), r.child;
            }
            i = i.sibling;
          }
          A.tail !== null && Fe() > Oc && (r.flags |= 128, d = !0, dd(A, !1), r.lanes = 4194304);
        }
        else {
          if (!d) if (i = xg(C), i !== null) {
            if (r.flags |= 128, d = !0, c = i.updateQueue, c !== null && (r.updateQueue = c, r.flags |= 4), dd(A, !0), A.tail === null && A.tailMode === "hidden" && !C.alternate && !ye) return En(r), null;
          } else 2 * Fe() - A.renderingStartTime > Oc && c !== 1073741824 && (r.flags |= 128, d = !0, dd(A, !1), r.lanes = 4194304);
          A.isBackwards ? (C.sibling = r.child, r.child = C) : (c = A.last, c !== null ? c.sibling = C : r.child = C, A.last = C);
        }
        return A.tail !== null ? (r = A.tail, A.rendering = r, A.tail = r.sibling, A.renderingStartTime = Fe(), r.sibling = null, c = Se.current, ue(Se, d ? c & 1 | 2 : c & 1), r) : (En(r), null);
      case 22:
      case 23:
        return c0(), d = r.memoizedState !== null, i !== null && i.memoizedState !== null !== d && (r.flags |= 8192), d && (r.mode & 1) !== 0 ? (ci & 1073741824) !== 0 && (En(r), r.subtreeFlags & 6 && (r.flags |= 8192)) : En(r), null;
      case 24:
        return null;
      case 25:
        return null;
    }
    throw Error(e(156, r.tag));
  }
  function vT(i, r) {
    switch (vv(r), r.tag) {
      case 1:
        return Wn(r.type) && gg(), i = r.flags, i & 65536 ? (r.flags = i & -65537 | 128, r) : null;
      case 3:
        return Dc(), ge(Vn), ge(Sn), Mv(), i = r.flags, (i & 65536) !== 0 && (i & 128) === 0 ? (r.flags = i & -65537 | 128, r) : null;
      case 5:
        return Pv(r), null;
      case 13:
        if (ge(Se), i = r.memoizedState, i !== null && i.dehydrated !== null) {
          if (r.alternate === null) throw Error(e(340));
          Pc();
        }
        return i = r.flags, i & 65536 ? (r.flags = i & -65537 | 128, r) : null;
      case 19:
        return ge(Se), null;
      case 4:
        return Dc(), null;
      case 10:
        return _v(r.type._context), null;
      case 22:
      case 23:
        return c0(), null;
      case 24:
        return null;
      default:
        return null;
    }
  }
  var Fg = !1, _n = !1, wT = typeof WeakSet == "function" ? WeakSet : Set, pt = null;
  function Fc(i, r) {
    var c = i.ref;
    if (c !== null) if (typeof c == "function") try {
      c(null);
    } catch (d) {
      ke(i, r, d);
    }
    else c.current = null;
  }
  function Qv(i, r, c) {
    try {
      c();
    } catch (d) {
      ke(i, r, d);
    }
  }
  var jb = !1;
  function AT(i, r) {
    if (cv = eg, i = AS(), ev(i)) {
      if ("selectionStart" in i) var c = { start: i.selectionStart, end: i.selectionEnd };
      else t: {
        c = (c = i.ownerDocument) && c.defaultView || window;
        var d = c.getSelection && c.getSelection();
        if (d && d.rangeCount !== 0) {
          c = d.anchorNode;
          var m = d.anchorOffset, A = d.focusNode;
          d = d.focusOffset;
          try {
            c.nodeType, A.nodeType;
          } catch {
            c = null;
            break t;
          }
          var C = 0, M = -1, F = -1, j = 0, Y = 0, J = i, G = null;
          e: for (; ; ) {
            for (var ut; J !== c || m !== 0 && J.nodeType !== 3 || (M = C + m), J !== A || d !== 0 && J.nodeType !== 3 || (F = C + d), J.nodeType === 3 && (C += J.nodeValue.length), (ut = J.firstChild) !== null; )
              G = J, J = ut;
            for (; ; ) {
              if (J === i) break e;
              if (G === c && ++j === m && (M = C), G === A && ++Y === d && (F = C), (ut = J.nextSibling) !== null) break;
              J = G, G = J.parentNode;
            }
            J = ut;
          }
          c = M === -1 || F === -1 ? null : { start: M, end: F };
        } else c = null;
      }
      c = c || { start: 0, end: 0 };
    } else c = null;
    for (uv = { focusedElem: i, selectionRange: c }, eg = !1, pt = r; pt !== null; ) if (r = pt, i = r.child, (r.subtreeFlags & 1028) !== 0 && i !== null) i.return = r, pt = i;
    else for (; pt !== null; ) {
      r = pt;
      try {
        var mt = r.alternate;
        if ((r.flags & 1024) !== 0) switch (r.tag) {
          case 0:
          case 11:
          case 15:
            break;
          case 1:
            if (mt !== null) {
              var yt = mt.memoizedProps, Ne = mt.memoizedState, U = r.stateNode, O = U.getSnapshotBeforeUpdate(r.elementType === r.type ? yt : cs(r.type, yt), Ne);
              U.__reactInternalSnapshotBeforeUpdate = O;
            }
            break;
          case 3:
            var H = r.stateNode.containerInfo;
            H.nodeType === 1 ? H.textContent = "" : H.nodeType === 9 && H.documentElement && H.removeChild(H.documentElement);
            break;
          case 5:
          case 6:
          case 4:
          case 17:
            break;
          default:
            throw Error(e(163));
        }
      } catch (et) {
        ke(r, r.return, et);
      }
      if (i = r.sibling, i !== null) {
        i.return = r.return, pt = i;
        break;
      }
      pt = r.return;
    }
    return mt = jb, jb = !1, mt;
  }
  function fd(i, r, c) {
    var d = r.updateQueue;
    if (d = d !== null ? d.lastEffect : null, d !== null) {
      var m = d = d.next;
      do {
        if ((m.tag & i) === i) {
          var A = m.destroy;
          m.destroy = void 0, A !== void 0 && Qv(r, c, A);
        }
        m = m.next;
      } while (m !== d);
    }
  }
  function Ng(i, r) {
    if (r = r.updateQueue, r = r !== null ? r.lastEffect : null, r !== null) {
      var c = r = r.next;
      do {
        if ((c.tag & i) === i) {
          var d = c.create;
          c.destroy = d();
        }
        c = c.next;
      } while (c !== r);
    }
  }
  function Zv(i) {
    var r = i.ref;
    if (r !== null) {
      var c = i.stateNode;
      switch (i.tag) {
        case 5:
          i = c;
          break;
        default:
          i = c;
      }
      typeof r == "function" ? r(i) : r.current = i;
    }
  }
  function $b(i) {
    var r = i.alternate;
    r !== null && (i.alternate = null, $b(r)), i.child = null, i.deletions = null, i.sibling = null, i.tag === 5 && (r = i.stateNode, r !== null && (delete r[Ls], delete r[nd], delete r[pv], delete r[nT], delete r[iT])), i.stateNode = null, i.return = null, i.dependencies = null, i.memoizedProps = null, i.memoizedState = null, i.pendingProps = null, i.stateNode = null, i.updateQueue = null;
  }
  function Vb(i) {
    return i.tag === 5 || i.tag === 3 || i.tag === 4;
  }
  function Wb(i) {
    t: for (; ; ) {
      for (; i.sibling === null; ) {
        if (i.return === null || Vb(i.return)) return null;
        i = i.return;
      }
      for (i.sibling.return = i.return, i = i.sibling; i.tag !== 5 && i.tag !== 6 && i.tag !== 18; ) {
        if (i.flags & 2 || i.child === null || i.tag === 4) continue t;
        i.child.return = i, i = i.child;
      }
      if (!(i.flags & 2)) return i.stateNode;
    }
  }
  function Jv(i, r, c) {
    var d = i.tag;
    if (d === 5 || d === 6) i = i.stateNode, r ? c.nodeType === 8 ? c.parentNode.insertBefore(i, r) : c.insertBefore(i, r) : (c.nodeType === 8 ? (r = c.parentNode, r.insertBefore(i, c)) : (r = c, r.appendChild(i)), c = c._reactRootContainer, c != null || r.onclick !== null || (r.onclick = fg));
    else if (d !== 4 && (i = i.child, i !== null)) for (Jv(i, r, c), i = i.sibling; i !== null; ) Jv(i, r, c), i = i.sibling;
  }
  function t0(i, r, c) {
    var d = i.tag;
    if (d === 5 || d === 6) i = i.stateNode, r ? c.insertBefore(i, r) : c.appendChild(i);
    else if (d !== 4 && (i = i.child, i !== null)) for (t0(i, r, c), i = i.sibling; i !== null; ) t0(i, r, c), i = i.sibling;
  }
  var un = null, us = !1;
  function Co(i, r, c) {
    for (c = c.child; c !== null; ) Gb(i, r, c), c = c.sibling;
  }
  function Gb(i, r, c) {
    if (Ms && typeof Ms.onCommitFiberUnmount == "function") try {
      Ms.onCommitFiberUnmount(Yp, c);
    } catch {
    }
    switch (c.tag) {
      case 5:
        _n || Fc(c, r);
      case 6:
        var d = un, m = us;
        un = null, Co(i, r, c), un = d, us = m, un !== null && (us ? (i = un, c = c.stateNode, i.nodeType === 8 ? i.parentNode.removeChild(c) : i.removeChild(c)) : un.removeChild(c.stateNode));
        break;
      case 18:
        un !== null && (us ? (i = un, c = c.stateNode, i.nodeType === 8 ? fv(i.parentNode, c) : i.nodeType === 1 && fv(i, c), Wh(i)) : fv(un, c.stateNode));
        break;
      case 4:
        d = un, m = us, un = c.stateNode.containerInfo, us = !0, Co(i, r, c), un = d, us = m;
        break;
      case 0:
      case 11:
      case 14:
      case 15:
        if (!_n && (d = c.updateQueue, d !== null && (d = d.lastEffect, d !== null))) {
          m = d = d.next;
          do {
            var A = m, C = A.destroy;
            A = A.tag, C !== void 0 && ((A & 2) !== 0 || (A & 4) !== 0) && Qv(c, r, C), m = m.next;
          } while (m !== d);
        }
        Co(i, r, c);
        break;
      case 1:
        if (!_n && (Fc(c, r), d = c.stateNode, typeof d.componentWillUnmount == "function")) try {
          d.props = c.memoizedProps, d.state = c.memoizedState, d.componentWillUnmount();
        } catch (M) {
          ke(c, r, M);
        }
        Co(i, r, c);
        break;
      case 21:
        Co(i, r, c);
        break;
      case 22:
        c.mode & 1 ? (_n = (d = _n) || c.memoizedState !== null, Co(i, r, c), _n = d) : Co(i, r, c);
        break;
      default:
        Co(i, r, c);
    }
  }
  function Xb(i) {
    var r = i.updateQueue;
    if (r !== null) {
      i.updateQueue = null;
      var c = i.stateNode;
      c === null && (c = i.stateNode = new wT()), r.forEach(function(d) {
        var m = PT.bind(null, i, d);
        c.has(d) || (c.add(d), d.then(m, m));
      });
    }
  }
  function hs(i, r) {
    var c = r.deletions;
    if (c !== null) for (var d = 0; d < c.length; d++) {
      var m = c[d];
      try {
        var A = i, C = r, M = C;
        t: for (; M !== null; ) {
          switch (M.tag) {
            case 5:
              un = M.stateNode, us = !1;
              break t;
            case 3:
              un = M.stateNode.containerInfo, us = !0;
              break t;
            case 4:
              un = M.stateNode.containerInfo, us = !0;
              break t;
          }
          M = M.return;
        }
        if (un === null) throw Error(e(160));
        Gb(A, C, m), un = null, us = !1;
        var F = m.alternate;
        F !== null && (F.return = null), m.return = null;
      } catch (j) {
        ke(m, r, j);
      }
    }
    if (r.subtreeFlags & 12854) for (r = r.child; r !== null; ) qb(r, i), r = r.sibling;
  }
  function qb(i, r) {
    var c = i.alternate, d = i.flags;
    switch (i.tag) {
      case 0:
      case 11:
      case 14:
      case 15:
        if (hs(r, i), Fs(i), d & 4) {
          try {
            fd(3, i, i.return), Ng(3, i);
          } catch (yt) {
            ke(i, i.return, yt);
          }
          try {
            fd(5, i, i.return);
          } catch (yt) {
            ke(i, i.return, yt);
          }
        }
        break;
      case 1:
        hs(r, i), Fs(i), d & 512 && c !== null && Fc(c, c.return);
        break;
      case 5:
        if (hs(r, i), Fs(i), d & 512 && c !== null && Fc(c, c.return), i.flags & 32) {
          var m = i.stateNode;
          try {
            hr(m, "");
          } catch (yt) {
            ke(i, i.return, yt);
          }
        }
        if (d & 4 && (m = i.stateNode, m != null)) {
          var A = i.memoizedProps, C = c !== null ? c.memoizedProps : A, M = i.type, F = i.updateQueue;
          if (i.updateQueue = null, F !== null) try {
            M === "input" && A.type === "radio" && A.name != null && Ie(m, A), My(M, C);
            var j = My(M, A);
            for (C = 0; C < F.length; C += 2) {
              var Y = F[C], J = F[C + 1];
              Y === "style" ? RA(m, J) : Y === "dangerouslySetInnerHTML" ? rs(m, J) : Y === "children" ? hr(m, J) : P(m, Y, J, j);
            }
            switch (M) {
              case "input":
                Rs(m, A);
                break;
              case "textarea":
                $p(m, A);
                break;
              case "select":
                var G = m._wrapperState.wasMultiple;
                m._wrapperState.wasMultiple = !!A.multiple;
                var ut = A.value;
                ut != null ? is(m, !!A.multiple, ut, !1) : G !== !!A.multiple && (A.defaultValue != null ? is(
                  m,
                  !!A.multiple,
                  A.defaultValue,
                  !0
                ) : is(m, !!A.multiple, A.multiple ? [] : "", !1));
            }
            m[nd] = A;
          } catch (yt) {
            ke(i, i.return, yt);
          }
        }
        break;
      case 6:
        if (hs(r, i), Fs(i), d & 4) {
          if (i.stateNode === null) throw Error(e(162));
          m = i.stateNode, A = i.memoizedProps;
          try {
            m.nodeValue = A;
          } catch (yt) {
            ke(i, i.return, yt);
          }
        }
        break;
      case 3:
        if (hs(r, i), Fs(i), d & 4 && c !== null && c.memoizedState.isDehydrated) try {
          Wh(r.containerInfo);
        } catch (yt) {
          ke(i, i.return, yt);
        }
        break;
      case 4:
        hs(r, i), Fs(i);
        break;
      case 13:
        hs(r, i), Fs(i), m = i.child, m.flags & 8192 && (A = m.memoizedState !== null, m.stateNode.isHidden = A, !A || m.alternate !== null && m.alternate.memoizedState !== null || (i0 = Fe())), d & 4 && Xb(i);
        break;
      case 22:
        if (Y = c !== null && c.memoizedState !== null, i.mode & 1 ? (_n = (j = _n) || Y, hs(r, i), _n = j) : hs(r, i), Fs(i), d & 8192) {
          if (j = i.memoizedState !== null, (i.stateNode.isHidden = j) && !Y && (i.mode & 1) !== 0) for (pt = i, Y = i.child; Y !== null; ) {
            for (J = pt = Y; pt !== null; ) {
              switch (G = pt, ut = G.child, G.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  fd(4, G, G.return);
                  break;
                case 1:
                  Fc(G, G.return);
                  var mt = G.stateNode;
                  if (typeof mt.componentWillUnmount == "function") {
                    d = G, c = G.return;
                    try {
                      r = d, mt.props = r.memoizedProps, mt.state = r.memoizedState, mt.componentWillUnmount();
                    } catch (yt) {
                      ke(d, c, yt);
                    }
                  }
                  break;
                case 5:
                  Fc(G, G.return);
                  break;
                case 22:
                  if (G.memoizedState !== null) {
                    Qb(J);
                    continue;
                  }
              }
              ut !== null ? (ut.return = G, pt = ut) : Qb(J);
            }
            Y = Y.sibling;
          }
          t: for (Y = null, J = i; ; ) {
            if (J.tag === 5) {
              if (Y === null) {
                Y = J;
                try {
                  m = J.stateNode, j ? (A = m.style, typeof A.setProperty == "function" ? A.setProperty("display", "none", "important") : A.display = "none") : (M = J.stateNode, F = J.memoizedProps.style, C = F != null && F.hasOwnProperty("display") ? F.display : null, M.style.display = PA("display", C));
                } catch (yt) {
                  ke(i, i.return, yt);
                }
              }
            } else if (J.tag === 6) {
              if (Y === null) try {
                J.stateNode.nodeValue = j ? "" : J.memoizedProps;
              } catch (yt) {
                ke(i, i.return, yt);
              }
            } else if ((J.tag !== 22 && J.tag !== 23 || J.memoizedState === null || J === i) && J.child !== null) {
              J.child.return = J, J = J.child;
              continue;
            }
            if (J === i) break t;
            for (; J.sibling === null; ) {
              if (J.return === null || J.return === i) break t;
              Y === J && (Y = null), J = J.return;
            }
            Y === J && (Y = null), J.sibling.return = J.return, J = J.sibling;
          }
        }
        break;
      case 19:
        hs(r, i), Fs(i), d & 4 && Xb(i);
        break;
      case 21:
        break;
      default:
        hs(
          r,
          i
        ), Fs(i);
    }
  }
  function Fs(i) {
    var r = i.flags;
    if (r & 2) {
      try {
        t: {
          for (var c = i.return; c !== null; ) {
            if (Vb(c)) {
              var d = c;
              break t;
            }
            c = c.return;
          }
          throw Error(e(160));
        }
        switch (d.tag) {
          case 5:
            var m = d.stateNode;
            d.flags & 32 && (hr(m, ""), d.flags &= -33);
            var A = Wb(i);
            t0(i, A, m);
            break;
          case 3:
          case 4:
            var C = d.stateNode.containerInfo, M = Wb(i);
            Jv(i, M, C);
            break;
          default:
            throw Error(e(161));
        }
      } catch (F) {
        ke(i, i.return, F);
      }
      i.flags &= -3;
    }
    r & 4096 && (i.flags &= -4097);
  }
  function ST(i, r, c) {
    pt = i, Yb(i);
  }
  function Yb(i, r, c) {
    for (var d = (i.mode & 1) !== 0; pt !== null; ) {
      var m = pt, A = m.child;
      if (m.tag === 22 && d) {
        var C = m.memoizedState !== null || Fg;
        if (!C) {
          var M = m.alternate, F = M !== null && M.memoizedState !== null || _n;
          M = Fg;
          var j = _n;
          if (Fg = C, (_n = F) && !j) for (pt = m; pt !== null; ) C = pt, F = C.child, C.tag === 22 && C.memoizedState !== null ? Zb(m) : F !== null ? (F.return = C, pt = F) : Zb(m);
          for (; A !== null; ) pt = A, Yb(A), A = A.sibling;
          pt = m, Fg = M, _n = j;
        }
        Kb(i);
      } else (m.subtreeFlags & 8772) !== 0 && A !== null ? (A.return = m, pt = A) : Kb(i);
    }
  }
  function Kb(i) {
    for (; pt !== null; ) {
      var r = pt;
      if ((r.flags & 8772) !== 0) {
        var c = r.alternate;
        try {
          if ((r.flags & 8772) !== 0) switch (r.tag) {
            case 0:
            case 11:
            case 15:
              _n || Ng(5, r);
              break;
            case 1:
              var d = r.stateNode;
              if (r.flags & 4 && !_n) if (c === null) d.componentDidMount();
              else {
                var m = r.elementType === r.type ? c.memoizedProps : cs(r.type, c.memoizedProps);
                d.componentDidUpdate(m, c.memoizedState, d.__reactInternalSnapshotBeforeUpdate);
              }
              var A = r.updateQueue;
              A !== null && QS(r, A, d);
              break;
            case 3:
              var C = r.updateQueue;
              if (C !== null) {
                if (c = null, r.child !== null) switch (r.child.tag) {
                  case 5:
                    c = r.child.stateNode;
                    break;
                  case 1:
                    c = r.child.stateNode;
                }
                QS(r, C, c);
              }
              break;
            case 5:
              var M = r.stateNode;
              if (c === null && r.flags & 4) {
                c = M;
                var F = r.memoizedProps;
                switch (r.type) {
                  case "button":
                  case "input":
                  case "select":
                  case "textarea":
                    F.autoFocus && c.focus();
                    break;
                  case "img":
                    F.src && (c.src = F.src);
                }
              }
              break;
            case 6:
              break;
            case 4:
              break;
            case 12:
              break;
            case 13:
              if (r.memoizedState === null) {
                var j = r.alternate;
                if (j !== null) {
                  var Y = j.memoizedState;
                  if (Y !== null) {
                    var J = Y.dehydrated;
                    J !== null && Wh(J);
                  }
                }
              }
              break;
            case 19:
            case 17:
            case 21:
            case 22:
            case 23:
            case 25:
              break;
            default:
              throw Error(e(163));
          }
          _n || r.flags & 512 && Zv(r);
        } catch (G) {
          ke(r, r.return, G);
        }
      }
      if (r === i) {
        pt = null;
        break;
      }
      if (c = r.sibling, c !== null) {
        c.return = r.return, pt = c;
        break;
      }
      pt = r.return;
    }
  }
  function Qb(i) {
    for (; pt !== null; ) {
      var r = pt;
      if (r === i) {
        pt = null;
        break;
      }
      var c = r.sibling;
      if (c !== null) {
        c.return = r.return, pt = c;
        break;
      }
      pt = r.return;
    }
  }
  function Zb(i) {
    for (; pt !== null; ) {
      var r = pt;
      try {
        switch (r.tag) {
          case 0:
          case 11:
          case 15:
            var c = r.return;
            try {
              Ng(4, r);
            } catch (F) {
              ke(r, c, F);
            }
            break;
          case 1:
            var d = r.stateNode;
            if (typeof d.componentDidMount == "function") {
              var m = r.return;
              try {
                d.componentDidMount();
              } catch (F) {
                ke(r, m, F);
              }
            }
            var A = r.return;
            try {
              Zv(r);
            } catch (F) {
              ke(r, A, F);
            }
            break;
          case 5:
            var C = r.return;
            try {
              Zv(r);
            } catch (F) {
              ke(r, C, F);
            }
        }
      } catch (F) {
        ke(r, r.return, F);
      }
      if (r === i) {
        pt = null;
        break;
      }
      var M = r.sibling;
      if (M !== null) {
        M.return = r.return, pt = M;
        break;
      }
      pt = r.return;
    }
  }
  var bT = Math.ceil, Og = L.ReactCurrentDispatcher, e0 = L.ReactCurrentOwner, zi = L.ReactCurrentBatchConfig, Vt = 0, Je = null, Ue = null, hn = 0, ci = 0, Nc = Ao(0), We = 0, pd = null, Ha = 0, Bg = 0, n0 = 0, gd = null, Xn = null, i0 = 0, Oc = 1 / 0, wr = null, zg = !1, s0 = null, xo = null, Ug = !1, To = null, Hg = 0, md = 0, r0 = null, jg = -1, $g = 0;
  function Fn() {
    return (Vt & 6) !== 0 ? Fe() : jg !== -1 ? jg : jg = Fe();
  }
  function ko(i) {
    return (i.mode & 1) === 0 ? 1 : (Vt & 2) !== 0 && hn !== 0 ? hn & -hn : rT.transition !== null ? ($g === 0 && ($g = WA()), $g) : (i = Kt, i !== 0 || (i = window.event, i = i === void 0 ? 16 : tS(i.type)), i);
  }
  function ds(i, r, c, d) {
    if (50 < md) throw md = 0, r0 = null, Error(e(185));
    Uh(i, c, d), ((Vt & 2) === 0 || i !== Je) && (i === Je && ((Vt & 2) === 0 && (Bg |= c), We === 4 && Po(i, hn)), qn(i, d), c === 1 && Vt === 0 && (r.mode & 1) === 0 && (Oc = Fe() + 500, yg && bo()));
  }
  function qn(i, r) {
    var c = i.callbackNode;
    rx(i, r);
    var d = Zp(i, i === Je ? hn : 0);
    if (d === 0) c !== null && jA(c), i.callbackNode = null, i.callbackPriority = 0;
    else if (r = d & -d, i.callbackPriority !== r) {
      if (c != null && jA(c), r === 1) i.tag === 0 ? sT(tE.bind(null, i)) : zS(tE.bind(null, i)), tT(function() {
        (Vt & 6) === 0 && bo();
      }), c = null;
      else {
        switch (GA(d)) {
          case 1:
            c = By;
            break;
          case 4:
            c = $A;
            break;
          case 16:
            c = qp;
            break;
          case 536870912:
            c = VA;
            break;
          default:
            c = qp;
        }
        c = lE(c, Jb.bind(null, i));
      }
      i.callbackPriority = r, i.callbackNode = c;
    }
  }
  function Jb(i, r) {
    if (jg = -1, $g = 0, (Vt & 6) !== 0) throw Error(e(327));
    var c = i.callbackNode;
    if (Bc() && i.callbackNode !== c) return null;
    var d = Zp(i, i === Je ? hn : 0);
    if (d === 0) return null;
    if ((d & 30) !== 0 || (d & i.expiredLanes) !== 0 || r) r = Vg(i, d);
    else {
      r = d;
      var m = Vt;
      Vt |= 2;
      var A = nE();
      (Je !== i || hn !== r) && (wr = null, Oc = Fe() + 500, $a(i, r));
      do
        try {
          CT();
          break;
        } catch (M) {
          eE(i, M);
        }
      while (!0);
      Ev(), Og.current = A, Vt = m, Ue !== null ? r = 0 : (Je = null, hn = 0, r = We);
    }
    if (r !== 0) {
      if (r === 2 && (m = zy(i), m !== 0 && (d = m, r = o0(i, m))), r === 1) throw c = pd, $a(i, 0), Po(i, d), qn(i, Fe()), c;
      if (r === 6) Po(i, d);
      else {
        if (m = i.current.alternate, (d & 30) === 0 && !ET(m) && (r = Vg(i, d), r === 2 && (A = zy(i), A !== 0 && (d = A, r = o0(i, A))), r === 1)) throw c = pd, $a(i, 0), Po(i, d), qn(i, Fe()), c;
        switch (i.finishedWork = m, i.finishedLanes = d, r) {
          case 0:
          case 1:
            throw Error(e(345));
          case 2:
            Va(i, Xn, wr);
            break;
          case 3:
            if (Po(i, d), (d & 130023424) === d && (r = i0 + 500 - Fe(), 10 < r)) {
              if (Zp(i, 0) !== 0) break;
              if (m = i.suspendedLanes, (m & d) !== d) {
                Fn(), i.pingedLanes |= i.suspendedLanes & m;
                break;
              }
              i.timeoutHandle = dv(Va.bind(null, i, Xn, wr), r);
              break;
            }
            Va(i, Xn, wr);
            break;
          case 4:
            if (Po(i, d), (d & 4194240) === d) break;
            for (r = i.eventTimes, m = -1; 0 < d; ) {
              var C = 31 - os(d);
              A = 1 << C, C = r[C], C > m && (m = C), d &= ~A;
            }
            if (d = m, d = Fe() - d, d = (120 > d ? 120 : 480 > d ? 480 : 1080 > d ? 1080 : 1920 > d ? 1920 : 3e3 > d ? 3e3 : 4320 > d ? 4320 : 1960 * bT(d / 1960)) - d, 10 < d) {
              i.timeoutHandle = dv(Va.bind(null, i, Xn, wr), d);
              break;
            }
            Va(i, Xn, wr);
            break;
          case 5:
            Va(i, Xn, wr);
            break;
          default:
            throw Error(e(329));
        }
      }
    }
    return qn(i, Fe()), i.callbackNode === c ? Jb.bind(null, i) : null;
  }
  function o0(i, r) {
    var c = gd;
    return i.current.memoizedState.isDehydrated && ($a(i, r).flags |= 256), i = Vg(i, r), i !== 2 && (r = Xn, Xn = c, r !== null && a0(r)), i;
  }
  function a0(i) {
    Xn === null ? Xn = i : Xn.push.apply(Xn, i);
  }
  function ET(i) {
    for (var r = i; ; ) {
      if (r.flags & 16384) {
        var c = r.updateQueue;
        if (c !== null && (c = c.stores, c !== null)) for (var d = 0; d < c.length; d++) {
          var m = c[d], A = m.getSnapshot;
          m = m.value;
          try {
            if (!as(A(), m)) return !1;
          } catch {
            return !1;
          }
        }
      }
      if (c = r.child, r.subtreeFlags & 16384 && c !== null) c.return = r, r = c;
      else {
        if (r === i) break;
        for (; r.sibling === null; ) {
          if (r.return === null || r.return === i) return !0;
          r = r.return;
        }
        r.sibling.return = r.return, r = r.sibling;
      }
    }
    return !0;
  }
  function Po(i, r) {
    for (r &= ~n0, r &= ~Bg, i.suspendedLanes |= r, i.pingedLanes &= ~r, i = i.expirationTimes; 0 < r; ) {
      var c = 31 - os(r), d = 1 << c;
      i[c] = -1, r &= ~d;
    }
  }
  function tE(i) {
    if ((Vt & 6) !== 0) throw Error(e(327));
    Bc();
    var r = Zp(i, 0);
    if ((r & 1) === 0) return qn(i, Fe()), null;
    var c = Vg(i, r);
    if (i.tag !== 0 && c === 2) {
      var d = zy(i);
      d !== 0 && (r = d, c = o0(i, d));
    }
    if (c === 1) throw c = pd, $a(i, 0), Po(i, r), qn(i, Fe()), c;
    if (c === 6) throw Error(e(345));
    return i.finishedWork = i.current.alternate, i.finishedLanes = r, Va(i, Xn, wr), qn(i, Fe()), null;
  }
  function l0(i, r) {
    var c = Vt;
    Vt |= 1;
    try {
      return i(r);
    } finally {
      Vt = c, Vt === 0 && (Oc = Fe() + 500, yg && bo());
    }
  }
  function ja(i) {
    To !== null && To.tag === 0 && (Vt & 6) === 0 && Bc();
    var r = Vt;
    Vt |= 1;
    var c = zi.transition, d = Kt;
    try {
      if (zi.transition = null, Kt = 1, i) return i();
    } finally {
      Kt = d, zi.transition = c, Vt = r, (Vt & 6) === 0 && bo();
    }
  }
  function c0() {
    ci = Nc.current, ge(Nc);
  }
  function $a(i, r) {
    i.finishedWork = null, i.finishedLanes = 0;
    var c = i.timeoutHandle;
    if (c !== -1 && (i.timeoutHandle = -1, Jx(c)), Ue !== null) for (c = Ue.return; c !== null; ) {
      var d = c;
      switch (vv(d), d.tag) {
        case 1:
          d = d.type.childContextTypes, d != null && gg();
          break;
        case 3:
          Dc(), ge(Vn), ge(Sn), Mv();
          break;
        case 5:
          Pv(d);
          break;
        case 4:
          Dc();
          break;
        case 13:
          ge(Se);
          break;
        case 19:
          ge(Se);
          break;
        case 10:
          _v(d.type._context);
          break;
        case 22:
        case 23:
          c0();
      }
      c = c.return;
    }
    if (Je = i, Ue = i = Ro(i.current, null), hn = ci = r, We = 0, pd = null, n0 = Bg = Ha = 0, Xn = gd = null, Ba !== null) {
      for (r = 0; r < Ba.length; r++) if (c = Ba[r], d = c.interleaved, d !== null) {
        c.interleaved = null;
        var m = d.next, A = c.pending;
        if (A !== null) {
          var C = A.next;
          A.next = m, d.next = C;
        }
        c.pending = d;
      }
      Ba = null;
    }
    return i;
  }
  function eE(i, r) {
    do {
      var c = Ue;
      try {
        if (Ev(), Tg.current = Mg, kg) {
          for (var d = be.memoizedState; d !== null; ) {
            var m = d.queue;
            m !== null && (m.pending = null), d = d.next;
          }
          kg = !1;
        }
        if (Ua = 0, Ze = Ve = be = null, ld = !1, cd = 0, e0.current = null, c === null || c.return === null) {
          We = 1, pd = r, Ue = null;
          break;
        }
        t: {
          var A = i, C = c.return, M = c, F = r;
          if (r = hn, M.flags |= 32768, F !== null && typeof F == "object" && typeof F.then == "function") {
            var j = F, Y = M, J = Y.tag;
            if ((Y.mode & 1) === 0 && (J === 0 || J === 11 || J === 15)) {
              var G = Y.alternate;
              G ? (Y.updateQueue = G.updateQueue, Y.memoizedState = G.memoizedState, Y.lanes = G.lanes) : (Y.updateQueue = null, Y.memoizedState = null);
            }
            var ut = xb(C);
            if (ut !== null) {
              ut.flags &= -257, Tb(ut, C, M, A, r), ut.mode & 1 && Cb(A, j, r), r = ut, F = j;
              var mt = r.updateQueue;
              if (mt === null) {
                var yt = /* @__PURE__ */ new Set();
                yt.add(F), r.updateQueue = yt;
              } else mt.add(F);
              break t;
            } else {
              if ((r & 1) === 0) {
                Cb(A, j, r), u0();
                break t;
              }
              F = Error(e(426));
            }
          } else if (ye && M.mode & 1) {
            var Ne = xb(C);
            if (Ne !== null) {
              (Ne.flags & 65536) === 0 && (Ne.flags |= 256), Tb(Ne, C, M, A, r), Sv(Ic(F, M));
              break t;
            }
          }
          A = F = Ic(F, M), We !== 4 && (We = 2), gd === null ? gd = [A] : gd.push(A), A = C;
          do {
            switch (A.tag) {
              case 3:
                A.flags |= 65536, r &= -r, A.lanes |= r;
                var U = Eb(A, F, r);
                KS(A, U);
                break t;
              case 1:
                M = F;
                var O = A.type, H = A.stateNode;
                if ((A.flags & 128) === 0 && (typeof O.getDerivedStateFromError == "function" || H !== null && typeof H.componentDidCatch == "function" && (xo === null || !xo.has(H)))) {
                  A.flags |= 65536, r &= -r, A.lanes |= r;
                  var et = _b(A, M, r);
                  KS(A, et);
                  break t;
                }
            }
            A = A.return;
          } while (A !== null);
        }
        sE(c);
      } catch (At) {
        r = At, Ue === c && c !== null && (Ue = c = c.return);
        continue;
      }
      break;
    } while (!0);
  }
  function nE() {
    var i = Og.current;
    return Og.current = Mg, i === null ? Mg : i;
  }
  function u0() {
    (We === 0 || We === 3 || We === 2) && (We = 4), Je === null || (Ha & 268435455) === 0 && (Bg & 268435455) === 0 || Po(Je, hn);
  }
  function Vg(i, r) {
    var c = Vt;
    Vt |= 2;
    var d = nE();
    (Je !== i || hn !== r) && (wr = null, $a(i, r));
    do
      try {
        _T();
        break;
      } catch (m) {
        eE(i, m);
      }
    while (!0);
    if (Ev(), Vt = c, Og.current = d, Ue !== null) throw Error(e(261));
    return Je = null, hn = 0, We;
  }
  function _T() {
    for (; Ue !== null; ) iE(Ue);
  }
  function CT() {
    for (; Ue !== null && !KC(); ) iE(Ue);
  }
  function iE(i) {
    var r = aE(i.alternate, i, ci);
    i.memoizedProps = i.pendingProps, r === null ? sE(i) : Ue = r, e0.current = null;
  }
  function sE(i) {
    var r = i;
    do {
      var c = r.alternate;
      if (i = r.return, (r.flags & 32768) === 0) {
        if (c = yT(c, r, ci), c !== null) {
          Ue = c;
          return;
        }
      } else {
        if (c = vT(c, r), c !== null) {
          c.flags &= 32767, Ue = c;
          return;
        }
        if (i !== null) i.flags |= 32768, i.subtreeFlags = 0, i.deletions = null;
        else {
          We = 6, Ue = null;
          return;
        }
      }
      if (r = r.sibling, r !== null) {
        Ue = r;
        return;
      }
      Ue = r = i;
    } while (r !== null);
    We === 0 && (We = 5);
  }
  function Va(i, r, c) {
    var d = Kt, m = zi.transition;
    try {
      zi.transition = null, Kt = 1, xT(i, r, c, d);
    } finally {
      zi.transition = m, Kt = d;
    }
    return null;
  }
  function xT(i, r, c, d) {
    do
      Bc();
    while (To !== null);
    if ((Vt & 6) !== 0) throw Error(e(327));
    c = i.finishedWork;
    var m = i.finishedLanes;
    if (c === null) return null;
    if (i.finishedWork = null, i.finishedLanes = 0, c === i.current) throw Error(e(177));
    i.callbackNode = null, i.callbackPriority = 0;
    var A = c.lanes | c.childLanes;
    if (ox(i, A), i === Je && (Ue = Je = null, hn = 0), (c.subtreeFlags & 2064) === 0 && (c.flags & 2064) === 0 || Ug || (Ug = !0, lE(qp, function() {
      return Bc(), null;
    })), A = (c.flags & 15990) !== 0, (c.subtreeFlags & 15990) !== 0 || A) {
      A = zi.transition, zi.transition = null;
      var C = Kt;
      Kt = 1;
      var M = Vt;
      Vt |= 4, e0.current = null, AT(i, c), qb(c, i), Gx(uv), eg = !!cv, uv = cv = null, i.current = c, ST(c), QC(), Vt = M, Kt = C, zi.transition = A;
    } else i.current = c;
    if (Ug && (Ug = !1, To = i, Hg = m), A = i.pendingLanes, A === 0 && (xo = null), tx(c.stateNode), qn(i, Fe()), r !== null) for (d = i.onRecoverableError, c = 0; c < r.length; c++) m = r[c], d(m.value, { componentStack: m.stack, digest: m.digest });
    if (zg) throw zg = !1, i = s0, s0 = null, i;
    return (Hg & 1) !== 0 && i.tag !== 0 && Bc(), A = i.pendingLanes, (A & 1) !== 0 ? i === r0 ? md++ : (md = 0, r0 = i) : md = 0, bo(), null;
  }
  function Bc() {
    if (To !== null) {
      var i = GA(Hg), r = zi.transition, c = Kt;
      try {
        if (zi.transition = null, Kt = 16 > i ? 16 : i, To === null) var d = !1;
        else {
          if (i = To, To = null, Hg = 0, (Vt & 6) !== 0) throw Error(e(331));
          var m = Vt;
          for (Vt |= 4, pt = i.current; pt !== null; ) {
            var A = pt, C = A.child;
            if ((pt.flags & 16) !== 0) {
              var M = A.deletions;
              if (M !== null) {
                for (var F = 0; F < M.length; F++) {
                  var j = M[F];
                  for (pt = j; pt !== null; ) {
                    var Y = pt;
                    switch (Y.tag) {
                      case 0:
                      case 11:
                      case 15:
                        fd(8, Y, A);
                    }
                    var J = Y.child;
                    if (J !== null) J.return = Y, pt = J;
                    else for (; pt !== null; ) {
                      Y = pt;
                      var G = Y.sibling, ut = Y.return;
                      if ($b(Y), Y === j) {
                        pt = null;
                        break;
                      }
                      if (G !== null) {
                        G.return = ut, pt = G;
                        break;
                      }
                      pt = ut;
                    }
                  }
                }
                var mt = A.alternate;
                if (mt !== null) {
                  var yt = mt.child;
                  if (yt !== null) {
                    mt.child = null;
                    do {
                      var Ne = yt.sibling;
                      yt.sibling = null, yt = Ne;
                    } while (yt !== null);
                  }
                }
                pt = A;
              }
            }
            if ((A.subtreeFlags & 2064) !== 0 && C !== null) C.return = A, pt = C;
            else t: for (; pt !== null; ) {
              if (A = pt, (A.flags & 2048) !== 0) switch (A.tag) {
                case 0:
                case 11:
                case 15:
                  fd(9, A, A.return);
              }
              var U = A.sibling;
              if (U !== null) {
                U.return = A.return, pt = U;
                break t;
              }
              pt = A.return;
            }
          }
          var O = i.current;
          for (pt = O; pt !== null; ) {
            C = pt;
            var H = C.child;
            if ((C.subtreeFlags & 2064) !== 0 && H !== null) H.return = C, pt = H;
            else t: for (C = O; pt !== null; ) {
              if (M = pt, (M.flags & 2048) !== 0) try {
                switch (M.tag) {
                  case 0:
                  case 11:
                  case 15:
                    Ng(9, M);
                }
              } catch (At) {
                ke(M, M.return, At);
              }
              if (M === C) {
                pt = null;
                break t;
              }
              var et = M.sibling;
              if (et !== null) {
                et.return = M.return, pt = et;
                break t;
              }
              pt = M.return;
            }
          }
          if (Vt = m, bo(), Ms && typeof Ms.onPostCommitFiberRoot == "function") try {
            Ms.onPostCommitFiberRoot(Yp, i);
          } catch {
          }
          d = !0;
        }
        return d;
      } finally {
        Kt = c, zi.transition = r;
      }
    }
    return !1;
  }
  function rE(i, r, c) {
    r = Ic(c, r), r = Eb(i, r, 1), i = _o(i, r, 1), r = Fn(), i !== null && (Uh(i, 1, r), qn(i, r));
  }
  function ke(i, r, c) {
    if (i.tag === 3) rE(i, i, c);
    else for (; r !== null; ) {
      if (r.tag === 3) {
        rE(r, i, c);
        break;
      } else if (r.tag === 1) {
        var d = r.stateNode;
        if (typeof r.type.getDerivedStateFromError == "function" || typeof d.componentDidCatch == "function" && (xo === null || !xo.has(d))) {
          i = Ic(c, i), i = _b(r, i, 1), r = _o(r, i, 1), i = Fn(), r !== null && (Uh(r, 1, i), qn(r, i));
          break;
        }
      }
      r = r.return;
    }
  }
  function TT(i, r, c) {
    var d = i.pingCache;
    d !== null && d.delete(r), r = Fn(), i.pingedLanes |= i.suspendedLanes & c, Je === i && (hn & c) === c && (We === 4 || We === 3 && (hn & 130023424) === hn && 500 > Fe() - i0 ? $a(i, 0) : n0 |= c), qn(i, r);
  }
  function oE(i, r) {
    r === 0 && ((i.mode & 1) === 0 ? r = 1 : (r = Qp, Qp <<= 1, (Qp & 130023424) === 0 && (Qp = 4194304)));
    var c = Fn();
    i = mr(i, r), i !== null && (Uh(i, r, c), qn(i, c));
  }
  function kT(i) {
    var r = i.memoizedState, c = 0;
    r !== null && (c = r.retryLane), oE(i, c);
  }
  function PT(i, r) {
    var c = 0;
    switch (i.tag) {
      case 13:
        var d = i.stateNode, m = i.memoizedState;
        m !== null && (c = m.retryLane);
        break;
      case 19:
        d = i.stateNode;
        break;
      default:
        throw Error(e(314));
    }
    d !== null && d.delete(r), oE(i, c);
  }
  var aE;
  aE = function(i, r, c) {
    if (i !== null) if (i.memoizedProps !== r.pendingProps || Vn.current) Gn = !0;
    else {
      if ((i.lanes & c) === 0 && (r.flags & 128) === 0) return Gn = !1, mT(i, r, c);
      Gn = (i.flags & 131072) !== 0;
    }
    else Gn = !1, ye && (r.flags & 1048576) !== 0 && US(r, wg, r.index);
    switch (r.lanes = 0, r.tag) {
      case 2:
        var d = r.type;
        Ig(i, r), i = r.pendingProps;
        var m = xc(r, Sn.current);
        Lc(r, c), m = Iv(null, r, d, i, m, c);
        var A = Fv();
        return r.flags |= 1, typeof m == "object" && m !== null && typeof m.render == "function" && m.$$typeof === void 0 ? (r.tag = 1, r.memoizedState = null, r.updateQueue = null, Wn(d) ? (A = !0, mg(r)) : A = !1, r.memoizedState = m.state !== null && m.state !== void 0 ? m.state : null, Tv(r), m.updater = Lg, r.stateNode = m, m._reactInternals = r, Hv(r, d, i, c), r = Wv(null, r, d, !0, A, c)) : (r.tag = 0, ye && A && yv(r), In(null, r, m, c), r = r.child), r;
      case 16:
        d = r.elementType;
        t: {
          switch (Ig(i, r), i = r.pendingProps, m = d._init, d = m(d._payload), r.type = d, m = r.tag = MT(d), i = cs(d, i), m) {
            case 0:
              r = Vv(null, r, d, i, c);
              break t;
            case 1:
              r = Db(null, r, d, i, c);
              break t;
            case 11:
              r = kb(null, r, d, i, c);
              break t;
            case 14:
              r = Pb(null, r, d, cs(d.type, i), c);
              break t;
          }
          throw Error(e(
            306,
            d,
            ""
          ));
        }
        return r;
      case 0:
        return d = r.type, m = r.pendingProps, m = r.elementType === d ? m : cs(d, m), Vv(i, r, d, m, c);
      case 1:
        return d = r.type, m = r.pendingProps, m = r.elementType === d ? m : cs(d, m), Db(i, r, d, m, c);
      case 3:
        t: {
          if (Ib(r), i === null) throw Error(e(387));
          d = r.pendingProps, A = r.memoizedState, m = A.element, YS(i, r), Cg(r, d, null, c);
          var C = r.memoizedState;
          if (d = C.element, A.isDehydrated) if (A = { element: d, isDehydrated: !1, cache: C.cache, pendingSuspenseBoundaries: C.pendingSuspenseBoundaries, transitions: C.transitions }, r.updateQueue.baseState = A, r.memoizedState = A, r.flags & 256) {
            m = Ic(Error(e(423)), r), r = Fb(i, r, d, c, m);
            break t;
          } else if (d !== m) {
            m = Ic(Error(e(424)), r), r = Fb(i, r, d, c, m);
            break t;
          } else for (li = wo(r.stateNode.containerInfo.firstChild), ai = r, ye = !0, ls = null, c = XS(r, null, d, c), r.child = c; c; ) c.flags = c.flags & -3 | 4096, c = c.sibling;
          else {
            if (Pc(), d === m) {
              r = vr(i, r, c);
              break t;
            }
            In(i, r, d, c);
          }
          r = r.child;
        }
        return r;
      case 5:
        return ZS(r), i === null && Av(r), d = r.type, m = r.pendingProps, A = i !== null ? i.memoizedProps : null, C = m.children, hv(d, m) ? C = null : A !== null && hv(d, A) && (r.flags |= 32), Lb(i, r), In(i, r, C, c), r.child;
      case 6:
        return i === null && Av(r), null;
      case 13:
        return Nb(i, r, c);
      case 4:
        return kv(r, r.stateNode.containerInfo), d = r.pendingProps, i === null ? r.child = Rc(r, null, d, c) : In(i, r, d, c), r.child;
      case 11:
        return d = r.type, m = r.pendingProps, m = r.elementType === d ? m : cs(d, m), kb(i, r, d, m, c);
      case 7:
        return In(i, r, r.pendingProps, c), r.child;
      case 8:
        return In(i, r, r.pendingProps.children, c), r.child;
      case 12:
        return In(i, r, r.pendingProps.children, c), r.child;
      case 10:
        t: {
          if (d = r.type._context, m = r.pendingProps, A = r.memoizedProps, C = m.value, ue(bg, d._currentValue), d._currentValue = C, A !== null) if (as(A.value, C)) {
            if (A.children === m.children && !Vn.current) {
              r = vr(i, r, c);
              break t;
            }
          } else for (A = r.child, A !== null && (A.return = r); A !== null; ) {
            var M = A.dependencies;
            if (M !== null) {
              C = A.child;
              for (var F = M.firstContext; F !== null; ) {
                if (F.context === d) {
                  if (A.tag === 1) {
                    F = yr(-1, c & -c), F.tag = 2;
                    var j = A.updateQueue;
                    if (j !== null) {
                      j = j.shared;
                      var Y = j.pending;
                      Y === null ? F.next = F : (F.next = Y.next, Y.next = F), j.pending = F;
                    }
                  }
                  A.lanes |= c, F = A.alternate, F !== null && (F.lanes |= c), Cv(
                    A.return,
                    c,
                    r
                  ), M.lanes |= c;
                  break;
                }
                F = F.next;
              }
            } else if (A.tag === 10) C = A.type === r.type ? null : A.child;
            else if (A.tag === 18) {
              if (C = A.return, C === null) throw Error(e(341));
              C.lanes |= c, M = C.alternate, M !== null && (M.lanes |= c), Cv(C, c, r), C = A.sibling;
            } else C = A.child;
            if (C !== null) C.return = A;
            else for (C = A; C !== null; ) {
              if (C === r) {
                C = null;
                break;
              }
              if (A = C.sibling, A !== null) {
                A.return = C.return, C = A;
                break;
              }
              C = C.return;
            }
            A = C;
          }
          In(i, r, m.children, c), r = r.child;
        }
        return r;
      case 9:
        return m = r.type, d = r.pendingProps.children, Lc(r, c), m = Oi(m), d = d(m), r.flags |= 1, In(i, r, d, c), r.child;
      case 14:
        return d = r.type, m = cs(d, r.pendingProps), m = cs(d.type, m), Pb(i, r, d, m, c);
      case 15:
        return Rb(i, r, r.type, r.pendingProps, c);
      case 17:
        return d = r.type, m = r.pendingProps, m = r.elementType === d ? m : cs(d, m), Ig(i, r), r.tag = 1, Wn(d) ? (i = !0, mg(r)) : i = !1, Lc(r, c), Sb(r, d, m), Hv(r, d, m, c), Wv(null, r, d, !0, i, c);
      case 19:
        return Bb(i, r, c);
      case 22:
        return Mb(i, r, c);
    }
    throw Error(e(156, r.tag));
  };
  function lE(i, r) {
    return HA(i, r);
  }
  function RT(i, r, c, d) {
    this.tag = i, this.key = c, this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null, this.index = 0, this.ref = null, this.pendingProps = r, this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null, this.mode = d, this.subtreeFlags = this.flags = 0, this.deletions = null, this.childLanes = this.lanes = 0, this.alternate = null;
  }
  function Ui(i, r, c, d) {
    return new RT(i, r, c, d);
  }
  function h0(i) {
    return i = i.prototype, !(!i || !i.isReactComponent);
  }
  function MT(i) {
    if (typeof i == "function") return h0(i) ? 1 : 0;
    if (i != null) {
      if (i = i.$$typeof, i === it) return 11;
      if (i === vt) return 14;
    }
    return 2;
  }
  function Ro(i, r) {
    var c = i.alternate;
    return c === null ? (c = Ui(i.tag, r, i.key, i.mode), c.elementType = i.elementType, c.type = i.type, c.stateNode = i.stateNode, c.alternate = i, i.alternate = c) : (c.pendingProps = r, c.type = i.type, c.flags = 0, c.subtreeFlags = 0, c.deletions = null), c.flags = i.flags & 14680064, c.childLanes = i.childLanes, c.lanes = i.lanes, c.child = i.child, c.memoizedProps = i.memoizedProps, c.memoizedState = i.memoizedState, c.updateQueue = i.updateQueue, r = i.dependencies, c.dependencies = r === null ? null : { lanes: r.lanes, firstContext: r.firstContext }, c.sibling = i.sibling, c.index = i.index, c.ref = i.ref, c;
  }
  function Wg(i, r, c, d, m, A) {
    var C = 2;
    if (d = i, typeof i == "function") h0(i) && (C = 1);
    else if (typeof i == "string") C = 5;
    else t: switch (i) {
      case B:
        return Wa(c.children, m, A, r);
      case $:
        C = 8, m |= 8;
        break;
      case V:
        return i = Ui(12, c, r, m | 2), i.elementType = V, i.lanes = A, i;
      case lt:
        return i = Ui(13, c, r, m), i.elementType = lt, i.lanes = A, i;
      case rt:
        return i = Ui(19, c, r, m), i.elementType = rt, i.lanes = A, i;
      case Q:
        return Gg(c, m, A, r);
      default:
        if (typeof i == "object" && i !== null) switch (i.$$typeof) {
          case X:
            C = 10;
            break t;
          case q:
            C = 9;
            break t;
          case it:
            C = 11;
            break t;
          case vt:
            C = 14;
            break t;
          case tt:
            C = 16, d = null;
            break t;
        }
        throw Error(e(130, i == null ? i : typeof i, ""));
    }
    return r = Ui(C, c, r, m), r.elementType = i, r.type = d, r.lanes = A, r;
  }
  function Wa(i, r, c, d) {
    return i = Ui(7, i, d, r), i.lanes = c, i;
  }
  function Gg(i, r, c, d) {
    return i = Ui(22, i, d, r), i.elementType = Q, i.lanes = c, i.stateNode = { isHidden: !1 }, i;
  }
  function d0(i, r, c) {
    return i = Ui(6, i, null, r), i.lanes = c, i;
  }
  function f0(i, r, c) {
    return r = Ui(4, i.children !== null ? i.children : [], i.key, r), r.lanes = c, r.stateNode = { containerInfo: i.containerInfo, pendingChildren: null, implementation: i.implementation }, r;
  }
  function LT(i, r, c, d, m) {
    this.tag = r, this.containerInfo = i, this.finishedWork = this.pingCache = this.current = this.pendingChildren = null, this.timeoutHandle = -1, this.callbackNode = this.pendingContext = this.context = null, this.callbackPriority = 0, this.eventTimes = Uy(0), this.expirationTimes = Uy(-1), this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0, this.entanglements = Uy(0), this.identifierPrefix = d, this.onRecoverableError = m, this.mutableSourceEagerHydrationData = null;
  }
  function p0(i, r, c, d, m, A, C, M, F) {
    return i = new LT(i, r, c, M, F), r === 1 ? (r = 1, A === !0 && (r |= 8)) : r = 0, A = Ui(3, null, null, r), i.current = A, A.stateNode = i, A.memoizedState = { element: d, isDehydrated: c, cache: null, transitions: null, pendingSuspenseBoundaries: null }, Tv(A), i;
  }
  function DT(i, r, c) {
    var d = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return { $$typeof: N, key: d == null ? null : "" + d, children: i, containerInfo: r, implementation: c };
  }
  function cE(i) {
    if (!i) return So;
    i = i._reactInternals;
    t: {
      if (Da(i) !== i || i.tag !== 1) throw Error(e(170));
      var r = i;
      do {
        switch (r.tag) {
          case 3:
            r = r.stateNode.context;
            break t;
          case 1:
            if (Wn(r.type)) {
              r = r.stateNode.__reactInternalMemoizedMergedChildContext;
              break t;
            }
        }
        r = r.return;
      } while (r !== null);
      throw Error(e(171));
    }
    if (i.tag === 1) {
      var c = i.type;
      if (Wn(c)) return OS(i, c, r);
    }
    return r;
  }
  function uE(i, r, c, d, m, A, C, M, F) {
    return i = p0(c, d, !0, i, m, A, C, M, F), i.context = cE(null), c = i.current, d = Fn(), m = ko(c), A = yr(d, m), A.callback = r ?? null, _o(c, A, m), i.current.lanes = m, Uh(i, m, d), qn(i, d), i;
  }
  function Xg(i, r, c, d) {
    var m = r.current, A = Fn(), C = ko(m);
    return c = cE(c), r.context === null ? r.context = c : r.pendingContext = c, r = yr(A, C), r.payload = { element: i }, d = d === void 0 ? null : d, d !== null && (r.callback = d), i = _o(m, r, C), i !== null && (ds(i, m, C, A), _g(i, m, C)), C;
  }
  function qg(i) {
    if (i = i.current, !i.child) return null;
    switch (i.child.tag) {
      case 5:
        return i.child.stateNode;
      default:
        return i.child.stateNode;
    }
  }
  function hE(i, r) {
    if (i = i.memoizedState, i !== null && i.dehydrated !== null) {
      var c = i.retryLane;
      i.retryLane = c !== 0 && c < r ? c : r;
    }
  }
  function g0(i, r) {
    hE(i, r), (i = i.alternate) && hE(i, r);
  }
  function IT() {
    return null;
  }
  var dE = typeof reportError == "function" ? reportError : function(i) {
    console.error(i);
  };
  function m0(i) {
    this._internalRoot = i;
  }
  Yg.prototype.render = m0.prototype.render = function(i) {
    var r = this._internalRoot;
    if (r === null) throw Error(e(409));
    Xg(i, r, null, null);
  }, Yg.prototype.unmount = m0.prototype.unmount = function() {
    var i = this._internalRoot;
    if (i !== null) {
      this._internalRoot = null;
      var r = i.containerInfo;
      ja(function() {
        Xg(null, i, null, null);
      }), r[dr] = null;
    }
  };
  function Yg(i) {
    this._internalRoot = i;
  }
  Yg.prototype.unstable_scheduleHydration = function(i) {
    if (i) {
      var r = YA();
      i = { blockedOn: null, target: i, priority: r };
      for (var c = 0; c < mo.length && r !== 0 && r < mo[c].priority; c++) ;
      mo.splice(c, 0, i), c === 0 && ZA(i);
    }
  };
  function y0(i) {
    return !(!i || i.nodeType !== 1 && i.nodeType !== 9 && i.nodeType !== 11);
  }
  function Kg(i) {
    return !(!i || i.nodeType !== 1 && i.nodeType !== 9 && i.nodeType !== 11 && (i.nodeType !== 8 || i.nodeValue !== " react-mount-point-unstable "));
  }
  function fE() {
  }
  function FT(i, r, c, d, m) {
    if (m) {
      if (typeof d == "function") {
        var A = d;
        d = function() {
          var j = qg(C);
          A.call(j);
        };
      }
      var C = uE(r, d, i, 0, null, !1, !1, "", fE);
      return i._reactRootContainer = C, i[dr] = C.current, td(i.nodeType === 8 ? i.parentNode : i), ja(), C;
    }
    for (; m = i.lastChild; ) i.removeChild(m);
    if (typeof d == "function") {
      var M = d;
      d = function() {
        var j = qg(F);
        M.call(j);
      };
    }
    var F = p0(i, 0, !1, null, null, !1, !1, "", fE);
    return i._reactRootContainer = F, i[dr] = F.current, td(i.nodeType === 8 ? i.parentNode : i), ja(function() {
      Xg(r, F, c, d);
    }), F;
  }
  function Qg(i, r, c, d, m) {
    var A = c._reactRootContainer;
    if (A) {
      var C = A;
      if (typeof m == "function") {
        var M = m;
        m = function() {
          var F = qg(C);
          M.call(F);
        };
      }
      Xg(r, C, i, m);
    } else C = FT(c, r, i, m, d);
    return qg(C);
  }
  XA = function(i) {
    switch (i.tag) {
      case 3:
        var r = i.stateNode;
        if (r.current.memoizedState.isDehydrated) {
          var c = zh(r.pendingLanes);
          c !== 0 && (Hy(r, c | 1), qn(r, Fe()), (Vt & 6) === 0 && (Oc = Fe() + 500, bo()));
        }
        break;
      case 13:
        ja(function() {
          var d = mr(i, 1);
          if (d !== null) {
            var m = Fn();
            ds(d, i, 1, m);
          }
        }), g0(i, 1);
    }
  }, jy = function(i) {
    if (i.tag === 13) {
      var r = mr(i, 134217728);
      if (r !== null) {
        var c = Fn();
        ds(r, i, 134217728, c);
      }
      g0(i, 134217728);
    }
  }, qA = function(i) {
    if (i.tag === 13) {
      var r = ko(i), c = mr(i, r);
      if (c !== null) {
        var d = Fn();
        ds(c, i, r, d);
      }
      g0(i, r);
    }
  }, YA = function() {
    return Kt;
  }, KA = function(i, r) {
    var c = Kt;
    try {
      return Kt = i, r();
    } finally {
      Kt = c;
    }
  }, Iy = function(i, r, c) {
    switch (r) {
      case "input":
        if (Rs(i, c), r = c.name, c.type === "radio" && r != null) {
          for (c = i; c.parentNode; ) c = c.parentNode;
          for (c = c.querySelectorAll("input[name=" + JSON.stringify("" + r) + '][type="radio"]'), r = 0; r < c.length; r++) {
            var d = c[r];
            if (d !== i && d.form === i.form) {
              var m = pg(d);
              if (!m) throw Error(e(90));
              Di(d), Rs(d, m);
            }
          }
        }
        break;
      case "textarea":
        $p(i, c);
        break;
      case "select":
        r = c.value, r != null && is(i, !!c.multiple, r, !1);
    }
  }, IA = l0, FA = ja;
  var NT = { usingClientEntryPoint: !1, Events: [id, _c, pg, LA, DA, l0] }, yd = { findFiberByHostInstance: Ia, bundleType: 0, version: "18.3.1", rendererPackageName: "react-dom" }, OT = { bundleType: yd.bundleType, version: yd.version, rendererPackageName: yd.rendererPackageName, rendererConfig: yd.rendererConfig, overrideHookState: null, overrideHookStateDeletePath: null, overrideHookStateRenamePath: null, overrideProps: null, overridePropsDeletePath: null, overridePropsRenamePath: null, setErrorHandler: null, setSuspenseHandler: null, scheduleUpdate: null, currentDispatcherRef: L.ReactCurrentDispatcher, findHostInstanceByFiber: function(i) {
    return i = zA(i), i === null ? null : i.stateNode;
  }, findFiberByHostInstance: yd.findFiberByHostInstance || IT, findHostInstancesForRefresh: null, scheduleRefresh: null, scheduleRoot: null, setRefreshHandler: null, getCurrentFiber: null, reconcilerVersion: "18.3.1-next-f1338f8080-20240426" };
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u") {
    var Zg = __REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!Zg.isDisabled && Zg.supportsFiber) try {
      Yp = Zg.inject(OT), Ms = Zg;
    } catch {
    }
  }
  return Yn.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = NT, Yn.createPortal = function(i, r) {
    var c = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!y0(r)) throw Error(e(200));
    return DT(i, r, null, c);
  }, Yn.createRoot = function(i, r) {
    if (!y0(i)) throw Error(e(299));
    var c = !1, d = "", m = dE;
    return r != null && (r.unstable_strictMode === !0 && (c = !0), r.identifierPrefix !== void 0 && (d = r.identifierPrefix), r.onRecoverableError !== void 0 && (m = r.onRecoverableError)), r = p0(i, 1, !1, null, null, c, !1, d, m), i[dr] = r.current, td(i.nodeType === 8 ? i.parentNode : i), new m0(r);
  }, Yn.findDOMNode = function(i) {
    if (i == null) return null;
    if (i.nodeType === 1) return i;
    var r = i._reactInternals;
    if (r === void 0)
      throw typeof i.render == "function" ? Error(e(188)) : (i = Object.keys(i).join(","), Error(e(268, i)));
    return i = zA(r), i = i === null ? null : i.stateNode, i;
  }, Yn.flushSync = function(i) {
    return ja(i);
  }, Yn.hydrate = function(i, r, c) {
    if (!Kg(r)) throw Error(e(200));
    return Qg(null, i, r, !0, c);
  }, Yn.hydrateRoot = function(i, r, c) {
    if (!y0(i)) throw Error(e(405));
    var d = c != null && c.hydratedSources || null, m = !1, A = "", C = dE;
    if (c != null && (c.unstable_strictMode === !0 && (m = !0), c.identifierPrefix !== void 0 && (A = c.identifierPrefix), c.onRecoverableError !== void 0 && (C = c.onRecoverableError)), r = uE(r, null, i, 1, c ?? null, m, !1, A, C), i[dr] = r.current, td(i), d) for (i = 0; i < d.length; i++) c = d[i], m = c._getVersion, m = m(c._source), r.mutableSourceEagerHydrationData == null ? r.mutableSourceEagerHydrationData = [c, m] : r.mutableSourceEagerHydrationData.push(
      c,
      m
    );
    return new Yg(r);
  }, Yn.render = function(i, r, c) {
    if (!Kg(r)) throw Error(e(200));
    return Qg(null, i, r, !1, c);
  }, Yn.unmountComponentAtNode = function(i) {
    if (!Kg(i)) throw Error(e(40));
    return i._reactRootContainer ? (ja(function() {
      Qg(null, null, i, !1, function() {
        i._reactRootContainer = null, i[dr] = null;
      });
    }), !0) : !1;
  }, Yn.unstable_batchedUpdates = l0, Yn.unstable_renderSubtreeIntoContainer = function(i, r, c, d) {
    if (!Kg(c)) throw Error(e(200));
    if (i == null || i._reactInternals === void 0) throw Error(e(38));
    return Qg(i, r, c, !1, d);
  }, Yn.version = "18.3.1-next-f1338f8080-20240426", Yn;
}
var bE;
function m1() {
  if (bE) return S0.exports;
  bE = 1;
  function p() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function"))
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(p);
      } catch (t) {
        console.error(t);
      }
  }
  return p(), S0.exports = XT(), S0.exports;
}
var EE;
function qT() {
  if (EE) return Jg;
  EE = 1;
  var p = m1();
  return Jg.createRoot = p.createRoot, Jg.hydrateRoot = p.hydrateRoot, Jg;
}
var YT = qT(), I0 = m1();
function zc(p, t, e) {
  let n = e.initialDeps ?? [], s;
  function a() {
    var l, u, h, f;
    let g;
    e.key && ((l = e.debug) != null && l.call(e)) && (g = Date.now());
    const y = p();
    if (!(y.length !== n.length || y.some((x, _) => n[_] !== x)))
      return s;
    n = y;
    let S;
    if (e.key && ((u = e.debug) != null && u.call(e)) && (S = Date.now()), s = t(...y), e.key && ((h = e.debug) != null && h.call(e))) {
      const x = Math.round((Date.now() - g) * 100) / 100, _ = Math.round((Date.now() - S) * 100) / 100, k = _ / 16, T = (R, P) => {
        for (R = String(R); R.length < P; )
          R = " " + R;
        return R;
      };
      console.info(
        `%c⏱ ${T(_, 5)} /${T(x, 5)} ms`,
        `
            font-size: .6rem;
            font-weight: bold;
            color: hsl(${Math.max(
          0,
          Math.min(120 - 120 * k, 120)
        )}deg 100% 31%);`,
        e == null ? void 0 : e.key
      );
    }
    return (f = e == null ? void 0 : e.onChange) == null || f.call(e, s), s;
  }
  return a.updateDeps = (l) => {
    n = l;
  }, a;
}
function _E(p, t) {
  if (p === void 0)
    throw new Error("Unexpected undefined");
  return p;
}
const KT = (p, t) => Math.abs(p - t) < 1.01, QT = (p, t, e) => {
  let n;
  return function(...s) {
    p.clearTimeout(n), n = p.setTimeout(() => t.apply(this, s), e);
  };
}, CE = (p) => {
  const { offsetWidth: t, offsetHeight: e } = p;
  return { width: t, height: e };
}, ZT = (p) => p, JT = (p) => {
  const t = Math.max(p.startIndex - p.overscan, 0), e = Math.min(p.endIndex + p.overscan, p.count - 1), n = [];
  for (let s = t; s <= e; s++)
    n.push(s);
  return n;
}, tk = (p, t) => {
  const e = p.scrollElement;
  if (!e)
    return;
  const n = p.targetWindow;
  if (!n)
    return;
  const s = (l) => {
    const { width: u, height: h } = l;
    t({ width: Math.round(u), height: Math.round(h) });
  };
  if (s(CE(e)), !n.ResizeObserver)
    return () => {
    };
  const a = new n.ResizeObserver((l) => {
    const u = () => {
      const h = l[0];
      if (h != null && h.borderBoxSize) {
        const f = h.borderBoxSize[0];
        if (f) {
          s({ width: f.inlineSize, height: f.blockSize });
          return;
        }
      }
      s(CE(e));
    };
    p.options.useAnimationFrameWithResizeObserver ? requestAnimationFrame(u) : u();
  });
  return a.observe(e, { box: "border-box" }), () => {
    a.unobserve(e);
  };
}, xE = {
  passive: !0
}, TE = typeof window > "u" ? !0 : "onscrollend" in window, ek = (p, t) => {
  const e = p.scrollElement;
  if (!e)
    return;
  const n = p.targetWindow;
  if (!n)
    return;
  let s = 0;
  const a = p.options.useScrollendEvent && TE ? () => {
  } : QT(
    n,
    () => {
      t(s, !1);
    },
    p.options.isScrollingResetDelay
  ), l = (g) => () => {
    const { horizontal: y, isRtl: v } = p.options;
    s = y ? e.scrollLeft * (v && -1 || 1) : e.scrollTop, a(), t(s, g);
  }, u = l(!0), h = l(!1);
  h(), e.addEventListener("scroll", u, xE);
  const f = p.options.useScrollendEvent && TE;
  return f && e.addEventListener("scrollend", h, xE), () => {
    e.removeEventListener("scroll", u), f && e.removeEventListener("scrollend", h);
  };
}, nk = (p, t, e) => {
  if (t != null && t.borderBoxSize) {
    const n = t.borderBoxSize[0];
    if (n)
      return Math.round(
        n[e.options.horizontal ? "inlineSize" : "blockSize"]
      );
  }
  return p[e.options.horizontal ? "offsetWidth" : "offsetHeight"];
}, ik = (p, {
  adjustments: t = 0,
  behavior: e
}, n) => {
  var s, a;
  const l = p + t;
  (a = (s = n.scrollElement) == null ? void 0 : s.scrollTo) == null || a.call(s, {
    [n.options.horizontal ? "left" : "top"]: l,
    behavior: e
  });
};
class sk {
  constructor(t) {
    this.unsubs = [], this.scrollElement = null, this.targetWindow = null, this.isScrolling = !1, this.measurementsCache = [], this.itemSizeCache = /* @__PURE__ */ new Map(), this.pendingMeasuredCacheIndexes = [], this.scrollRect = null, this.scrollOffset = null, this.scrollDirection = null, this.scrollAdjustments = 0, this.elementsCache = /* @__PURE__ */ new Map(), this.observer = /* @__PURE__ */ (() => {
      let e = null;
      const n = () => e || (!this.targetWindow || !this.targetWindow.ResizeObserver ? null : e = new this.targetWindow.ResizeObserver((s) => {
        s.forEach((a) => {
          const l = () => {
            this._measureElement(a.target, a);
          };
          this.options.useAnimationFrameWithResizeObserver ? requestAnimationFrame(l) : l();
        });
      }));
      return {
        disconnect: () => {
          var s;
          (s = n()) == null || s.disconnect(), e = null;
        },
        observe: (s) => {
          var a;
          return (a = n()) == null ? void 0 : a.observe(s, { box: "border-box" });
        },
        unobserve: (s) => {
          var a;
          return (a = n()) == null ? void 0 : a.unobserve(s);
        }
      };
    })(), this.range = null, this.setOptions = (e) => {
      Object.entries(e).forEach(([n, s]) => {
        typeof s > "u" && delete e[n];
      }), this.options = {
        debug: !1,
        initialOffset: 0,
        overscan: 1,
        paddingStart: 0,
        paddingEnd: 0,
        scrollPaddingStart: 0,
        scrollPaddingEnd: 0,
        horizontal: !1,
        getItemKey: ZT,
        rangeExtractor: JT,
        onChange: () => {
        },
        measureElement: nk,
        initialRect: { width: 0, height: 0 },
        scrollMargin: 0,
        gap: 0,
        indexAttribute: "data-index",
        initialMeasurementsCache: [],
        lanes: 1,
        isScrollingResetDelay: 150,
        enabled: !0,
        isRtl: !1,
        useScrollendEvent: !1,
        useAnimationFrameWithResizeObserver: !1,
        ...e
      };
    }, this.notify = (e) => {
      var n, s;
      (s = (n = this.options).onChange) == null || s.call(n, this, e);
    }, this.maybeNotify = zc(
      () => (this.calculateRange(), [
        this.isScrolling,
        this.range ? this.range.startIndex : null,
        this.range ? this.range.endIndex : null
      ]),
      (e) => {
        this.notify(e);
      },
      {
        key: !1,
        debug: () => this.options.debug,
        initialDeps: [
          this.isScrolling,
          this.range ? this.range.startIndex : null,
          this.range ? this.range.endIndex : null
        ]
      }
    ), this.cleanup = () => {
      this.unsubs.filter(Boolean).forEach((e) => e()), this.unsubs = [], this.observer.disconnect(), this.scrollElement = null, this.targetWindow = null;
    }, this._didMount = () => () => {
      this.cleanup();
    }, this._willUpdate = () => {
      var e;
      const n = this.options.enabled ? this.options.getScrollElement() : null;
      if (this.scrollElement !== n) {
        if (this.cleanup(), !n) {
          this.maybeNotify();
          return;
        }
        this.scrollElement = n, this.scrollElement && "ownerDocument" in this.scrollElement ? this.targetWindow = this.scrollElement.ownerDocument.defaultView : this.targetWindow = ((e = this.scrollElement) == null ? void 0 : e.window) ?? null, this.elementsCache.forEach((s) => {
          this.observer.observe(s);
        }), this._scrollToOffset(this.getScrollOffset(), {
          adjustments: void 0,
          behavior: void 0
        }), this.unsubs.push(
          this.options.observeElementRect(this, (s) => {
            this.scrollRect = s, this.maybeNotify();
          })
        ), this.unsubs.push(
          this.options.observeElementOffset(this, (s, a) => {
            this.scrollAdjustments = 0, this.scrollDirection = a ? this.getScrollOffset() < s ? "forward" : "backward" : null, this.scrollOffset = s, this.isScrolling = a, this.maybeNotify();
          })
        );
      }
    }, this.getSize = () => this.options.enabled ? (this.scrollRect = this.scrollRect ?? this.options.initialRect, this.scrollRect[this.options.horizontal ? "width" : "height"]) : (this.scrollRect = null, 0), this.getScrollOffset = () => this.options.enabled ? (this.scrollOffset = this.scrollOffset ?? (typeof this.options.initialOffset == "function" ? this.options.initialOffset() : this.options.initialOffset), this.scrollOffset) : (this.scrollOffset = null, 0), this.getFurthestMeasurement = (e, n) => {
      const s = /* @__PURE__ */ new Map(), a = /* @__PURE__ */ new Map();
      for (let l = n - 1; l >= 0; l--) {
        const u = e[l];
        if (s.has(u.lane))
          continue;
        const h = a.get(
          u.lane
        );
        if (h == null || u.end > h.end ? a.set(u.lane, u) : u.end < h.end && s.set(u.lane, !0), s.size === this.options.lanes)
          break;
      }
      return a.size === this.options.lanes ? Array.from(a.values()).sort((l, u) => l.end === u.end ? l.index - u.index : l.end - u.end)[0] : void 0;
    }, this.getMeasurementOptions = zc(
      () => [
        this.options.count,
        this.options.paddingStart,
        this.options.scrollMargin,
        this.options.getItemKey,
        this.options.enabled
      ],
      (e, n, s, a, l) => (this.pendingMeasuredCacheIndexes = [], {
        count: e,
        paddingStart: n,
        scrollMargin: s,
        getItemKey: a,
        enabled: l
      }),
      {
        key: !1
      }
    ), this.getMeasurements = zc(
      () => [this.getMeasurementOptions(), this.itemSizeCache],
      ({ count: e, paddingStart: n, scrollMargin: s, getItemKey: a, enabled: l }, u) => {
        if (!l)
          return this.measurementsCache = [], this.itemSizeCache.clear(), [];
        this.measurementsCache.length === 0 && (this.measurementsCache = this.options.initialMeasurementsCache, this.measurementsCache.forEach((g) => {
          this.itemSizeCache.set(g.key, g.size);
        }));
        const h = this.pendingMeasuredCacheIndexes.length > 0 ? Math.min(...this.pendingMeasuredCacheIndexes) : 0;
        this.pendingMeasuredCacheIndexes = [];
        const f = this.measurementsCache.slice(0, h);
        for (let g = h; g < e; g++) {
          const y = a(g), v = this.options.lanes === 1 ? f[g - 1] : this.getFurthestMeasurement(f, g), S = v ? v.end + this.options.gap : n + s, x = u.get(y), _ = typeof x == "number" ? x : this.options.estimateSize(g), k = S + _, T = v ? v.lane : g % this.options.lanes;
          f[g] = {
            index: g,
            start: S,
            size: _,
            end: k,
            key: y,
            lane: T
          };
        }
        return this.measurementsCache = f, f;
      },
      {
        key: !1,
        debug: () => this.options.debug
      }
    ), this.calculateRange = zc(
      () => [
        this.getMeasurements(),
        this.getSize(),
        this.getScrollOffset(),
        this.options.lanes
      ],
      (e, n, s, a) => this.range = e.length > 0 && n > 0 ? rk({
        measurements: e,
        outerSize: n,
        scrollOffset: s,
        lanes: a
      }) : null,
      {
        key: !1,
        debug: () => this.options.debug
      }
    ), this.getVirtualIndexes = zc(
      () => {
        let e = null, n = null;
        const s = this.calculateRange();
        return s && (e = s.startIndex, n = s.endIndex), this.maybeNotify.updateDeps([this.isScrolling, e, n]), [
          this.options.rangeExtractor,
          this.options.overscan,
          this.options.count,
          e,
          n
        ];
      },
      (e, n, s, a, l) => a === null || l === null ? [] : e({
        startIndex: a,
        endIndex: l,
        overscan: n,
        count: s
      }),
      {
        key: !1,
        debug: () => this.options.debug
      }
    ), this.indexFromElement = (e) => {
      const n = this.options.indexAttribute, s = e.getAttribute(n);
      return s ? parseInt(s, 10) : (console.warn(
        `Missing attribute name '${n}={index}' on measured element.`
      ), -1);
    }, this._measureElement = (e, n) => {
      const s = this.indexFromElement(e), a = this.measurementsCache[s];
      if (!a)
        return;
      const l = a.key, u = this.elementsCache.get(l);
      u !== e && (u && this.observer.unobserve(u), this.observer.observe(e), this.elementsCache.set(l, e)), e.isConnected && this.resizeItem(s, this.options.measureElement(e, n, this));
    }, this.resizeItem = (e, n) => {
      const s = this.measurementsCache[e];
      if (!s)
        return;
      const a = this.itemSizeCache.get(s.key) ?? s.size, l = n - a;
      l !== 0 && ((this.shouldAdjustScrollPositionOnItemSizeChange !== void 0 ? this.shouldAdjustScrollPositionOnItemSizeChange(s, l, this) : s.start < this.getScrollOffset() + this.scrollAdjustments) && this._scrollToOffset(this.getScrollOffset(), {
        adjustments: this.scrollAdjustments += l,
        behavior: void 0
      }), this.pendingMeasuredCacheIndexes.push(s.index), this.itemSizeCache = new Map(this.itemSizeCache.set(s.key, n)), this.notify(!1));
    }, this.measureElement = (e) => {
      if (!e) {
        this.elementsCache.forEach((n, s) => {
          n.isConnected || (this.observer.unobserve(n), this.elementsCache.delete(s));
        });
        return;
      }
      this._measureElement(e, void 0);
    }, this.getVirtualItems = zc(
      () => [this.getVirtualIndexes(), this.getMeasurements()],
      (e, n) => {
        const s = [];
        for (let a = 0, l = e.length; a < l; a++) {
          const u = e[a], h = n[u];
          s.push(h);
        }
        return s;
      },
      {
        key: !1,
        debug: () => this.options.debug
      }
    ), this.getVirtualItemForOffset = (e) => {
      const n = this.getMeasurements();
      if (n.length !== 0)
        return _E(
          n[y1(
            0,
            n.length - 1,
            (s) => _E(n[s]).start,
            e
          )]
        );
    }, this.getOffsetForAlignment = (e, n, s = 0) => {
      const a = this.getSize(), l = this.getScrollOffset();
      n === "auto" && (n = e >= l + a ? "end" : "start"), n === "center" ? e += (s - a) / 2 : n === "end" && (e -= a);
      const u = this.getTotalSize() + this.options.scrollMargin - a;
      return Math.max(Math.min(u, e), 0);
    }, this.getOffsetForIndex = (e, n = "auto") => {
      e = Math.max(0, Math.min(e, this.options.count - 1));
      const s = this.measurementsCache[e];
      if (!s)
        return;
      const a = this.getSize(), l = this.getScrollOffset();
      if (n === "auto")
        if (s.end >= l + a - this.options.scrollPaddingEnd)
          n = "end";
        else if (s.start <= l + this.options.scrollPaddingStart)
          n = "start";
        else
          return [l, n];
      const u = n === "end" ? s.end + this.options.scrollPaddingEnd : s.start - this.options.scrollPaddingStart;
      return [
        this.getOffsetForAlignment(u, n, s.size),
        n
      ];
    }, this.isDynamicMode = () => this.elementsCache.size > 0, this.scrollToOffset = (e, { align: n = "start", behavior: s } = {}) => {
      s === "smooth" && this.isDynamicMode() && console.warn(
        "The `smooth` scroll behavior is not fully supported with dynamic size."
      ), this._scrollToOffset(this.getOffsetForAlignment(e, n), {
        adjustments: void 0,
        behavior: s
      });
    }, this.scrollToIndex = (e, { align: n = "auto", behavior: s } = {}) => {
      s === "smooth" && this.isDynamicMode() && console.warn(
        "The `smooth` scroll behavior is not fully supported with dynamic size."
      ), e = Math.max(0, Math.min(e, this.options.count - 1));
      let a = 0;
      const l = 10, u = (f) => {
        if (!this.targetWindow) return;
        const g = this.getOffsetForIndex(e, f);
        if (!g) {
          console.warn("Failed to get offset for index:", e);
          return;
        }
        const [y, v] = g;
        this._scrollToOffset(y, { adjustments: void 0, behavior: s }), this.targetWindow.requestAnimationFrame(() => {
          const S = this.getScrollOffset(), x = this.getOffsetForIndex(e, v);
          if (!x) {
            console.warn("Failed to get offset for index:", e);
            return;
          }
          KT(x[0], S) || h(v);
        });
      }, h = (f) => {
        this.targetWindow && (a++, a < l ? this.targetWindow.requestAnimationFrame(() => u(f)) : console.warn(
          `Failed to scroll to index ${e} after ${l} attempts.`
        ));
      };
      u(n);
    }, this.scrollBy = (e, { behavior: n } = {}) => {
      n === "smooth" && this.isDynamicMode() && console.warn(
        "The `smooth` scroll behavior is not fully supported with dynamic size."
      ), this._scrollToOffset(this.getScrollOffset() + e, {
        adjustments: void 0,
        behavior: n
      });
    }, this.getTotalSize = () => {
      var e;
      const n = this.getMeasurements();
      let s;
      if (n.length === 0)
        s = this.options.paddingStart;
      else if (this.options.lanes === 1)
        s = ((e = n[n.length - 1]) == null ? void 0 : e.end) ?? 0;
      else {
        const a = Array(this.options.lanes).fill(null);
        let l = n.length - 1;
        for (; l >= 0 && a.some((u) => u === null); ) {
          const u = n[l];
          a[u.lane] === null && (a[u.lane] = u.end), l--;
        }
        s = Math.max(...a.filter((u) => u !== null));
      }
      return Math.max(
        s - this.options.scrollMargin + this.options.paddingEnd,
        0
      );
    }, this._scrollToOffset = (e, {
      adjustments: n,
      behavior: s
    }) => {
      this.options.scrollToFn(e, { behavior: s, adjustments: n }, this);
    }, this.measure = () => {
      this.itemSizeCache = /* @__PURE__ */ new Map(), this.notify(!1);
    }, this.setOptions(t);
  }
}
const y1 = (p, t, e, n) => {
  for (; p <= t; ) {
    const s = (p + t) / 2 | 0, a = e(s);
    if (a < n)
      p = s + 1;
    else if (a > n)
      t = s - 1;
    else
      return s;
  }
  return p > 0 ? p - 1 : 0;
};
function rk({
  measurements: p,
  outerSize: t,
  scrollOffset: e,
  lanes: n
}) {
  const s = p.length - 1, a = (h) => p[h].start;
  if (p.length <= n)
    return {
      startIndex: 0,
      endIndex: s
    };
  let l = y1(
    0,
    s,
    a,
    e
  ), u = l;
  if (n === 1)
    for (; u < s && p[u].end < e + t; )
      u++;
  else if (n > 1) {
    const h = Array(n).fill(0);
    for (; u < s && h.some((g) => g < e + t); ) {
      const g = p[u];
      h[g.lane] = g.end, u++;
    }
    const f = Array(n).fill(e + t);
    for (; l >= 0 && f.some((g) => g >= e); ) {
      const g = p[l];
      f[g.lane] = g.start, l--;
    }
    l = Math.max(0, l - l % n), u = Math.min(s, u + (n - 1 - u % n));
  }
  return { startIndex: l, endIndex: u };
}
const kE = typeof document < "u" ? nt.useLayoutEffect : nt.useEffect;
function ok(p) {
  const t = nt.useReducer(() => ({}), {})[1], e = {
    ...p,
    onChange: (s, a) => {
      var l;
      a ? I0.flushSync(t) : t(), (l = p.onChange) == null || l.call(p, s, a);
    }
  }, [n] = nt.useState(
    () => new sk(e)
  );
  return n.setOptions(e), kE(() => n._didMount(), []), kE(() => n._willUpdate()), n;
}
function ak(p) {
  return ok({
    observeElementRect: tk,
    observeElementOffset: ek,
    scrollToFn: ik,
    ...p
  });
}
const Hn = typeof process == "object" && process + "" == "[object process]" && !process.versions.nw && !(process.versions.electron && process.type && process.type !== "browser"), F0 = [1e-3, 0, 0, 1e-3, 0, 0], _0 = 1.35, ki = {
  ANY: 1,
  DISPLAY: 2,
  PRINT: 4,
  ANNOTATIONS_FORMS: 16,
  ANNOTATIONS_STORAGE: 32,
  ANNOTATIONS_DISABLE: 64,
  IS_EDITING: 128,
  OPLIST: 256
}, lr = {
  DISABLE: 0,
  ENABLE: 1,
  ENABLE_FORMS: 2,
  ENABLE_STORAGE: 3
}, Wd = "pdfjs_internal_editor_", Pt = {
  DISABLE: -1,
  NONE: 0,
  FREETEXT: 3,
  HIGHLIGHT: 9,
  STAMP: 13,
  INK: 15,
  POPUP: 16,
  SIGNATURE: 101,
  COMMENT: 102
}, jt = {
  RESIZE: 1,
  CREATE: 2,
  FREETEXT_SIZE: 11,
  FREETEXT_COLOR: 12,
  FREETEXT_OPACITY: 13,
  INK_COLOR: 21,
  INK_THICKNESS: 22,
  INK_OPACITY: 23,
  HIGHLIGHT_COLOR: 31,
  HIGHLIGHT_THICKNESS: 32,
  HIGHLIGHT_FREE: 33,
  HIGHLIGHT_SHOW_ALL: 34,
  DRAW_STEP: 41
}, v1 = {
  PRINT: 4,
  MODIFY_CONTENTS: 8,
  COPY: 16,
  MODIFY_ANNOTATIONS: 32,
  FILL_INTERACTIVE_FORMS: 256,
  COPY_FOR_ACCESSIBILITY: 512,
  ASSEMBLE: 1024,
  PRINT_HIGH_QUALITY: 2048
}, dn = {
  FILL: 0,
  STROKE: 1,
  FILL_STROKE: 2,
  INVISIBLE: 3,
  FILL_STROKE_MASK: 3,
  ADD_TO_PATH_FLAG: 4
}, Vd = {
  GRAYSCALE_1BPP: 1,
  RGB_24BPP: 2,
  RGBA_32BPP: 3
}, Oe = {
  TEXT: 1,
  LINK: 2,
  FREETEXT: 3,
  LINE: 4,
  SQUARE: 5,
  CIRCLE: 6,
  POLYGON: 7,
  POLYLINE: 8,
  HIGHLIGHT: 9,
  UNDERLINE: 10,
  SQUIGGLY: 11,
  STRIKEOUT: 12,
  STAMP: 13,
  CARET: 14,
  INK: 15,
  POPUP: 16,
  FILEATTACHMENT: 17,
  SOUND: 18,
  MOVIE: 19,
  WIDGET: 20,
  SCREEN: 21,
  PRINTERMARK: 22,
  TRAPNET: 23,
  WATERMARK: 24,
  THREED: 25,
  REDACT: 26
}, Hc = {
  SOLID: 1,
  DASHED: 2,
  BEVELED: 3,
  INSET: 4,
  UNDERLINE: 5
}, Ip = {
  ERRORS: 0,
  WARNINGS: 1,
  INFOS: 5
}, Ph = {
  dependency: 1,
  setLineWidth: 2,
  setLineCap: 3,
  setLineJoin: 4,
  setMiterLimit: 5,
  setDash: 6,
  setRenderingIntent: 7,
  setFlatness: 8,
  setGState: 9,
  save: 10,
  restore: 11,
  transform: 12,
  moveTo: 13,
  lineTo: 14,
  curveTo: 15,
  curveTo2: 16,
  curveTo3: 17,
  closePath: 18,
  rectangle: 19,
  stroke: 20,
  closeStroke: 21,
  fill: 22,
  eoFill: 23,
  fillStroke: 24,
  eoFillStroke: 25,
  closeFillStroke: 26,
  closeEOFillStroke: 27,
  endPath: 28,
  clip: 29,
  eoClip: 30,
  beginText: 31,
  endText: 32,
  setCharSpacing: 33,
  setWordSpacing: 34,
  setHScale: 35,
  setLeading: 36,
  setFont: 37,
  setTextRenderingMode: 38,
  setTextRise: 39,
  moveText: 40,
  setLeadingMoveText: 41,
  setTextMatrix: 42,
  nextLine: 43,
  showText: 44,
  showSpacedText: 45,
  nextLineShowText: 46,
  nextLineSetSpacingShowText: 47,
  setCharWidth: 48,
  setCharWidthAndBounds: 49,
  setStrokeColorSpace: 50,
  setFillColorSpace: 51,
  setStrokeColor: 52,
  setStrokeColorN: 53,
  setFillColor: 54,
  setFillColorN: 55,
  setStrokeGray: 56,
  setFillGray: 57,
  setStrokeRGBColor: 58,
  setFillRGBColor: 59,
  setStrokeCMYKColor: 60,
  setFillCMYKColor: 61,
  shadingFill: 62,
  beginInlineImage: 63,
  beginImageData: 64,
  endInlineImage: 65,
  paintXObject: 66,
  markPoint: 67,
  markPointProps: 68,
  beginMarkedContent: 69,
  beginMarkedContentProps: 70,
  endMarkedContent: 71,
  beginCompat: 72,
  endCompat: 73,
  paintFormXObjectBegin: 74,
  paintFormXObjectEnd: 75,
  beginGroup: 76,
  endGroup: 77,
  beginAnnotation: 80,
  endAnnotation: 81,
  paintImageMaskXObject: 83,
  paintImageMaskXObjectGroup: 84,
  paintImageXObject: 85,
  paintInlineImageXObject: 86,
  paintInlineImageXObjectGroup: 87,
  paintImageXObjectRepeat: 88,
  paintImageMaskXObjectRepeat: 89,
  paintSolidColorImageMask: 90,
  constructPath: 91,
  setStrokeTransparent: 92,
  setFillTransparent: 93,
  rawFillPath: 94
}, tm = {
  moveTo: 0,
  lineTo: 1,
  curveTo: 2,
  closePath: 3
}, w1 = {
  NEED_PASSWORD: 1,
  INCORRECT_PASSWORD: 2
};
let Ay = Ip.WARNINGS;
function lk(p) {
  Number.isInteger(p) && (Ay = p);
}
function ck() {
  return Ay;
}
function Sy(p) {
  Ay >= Ip.INFOS && console.info(`Info: ${p}`);
}
function Rt(p) {
  Ay >= Ip.WARNINGS && console.warn(`Warning: ${p}`);
}
function ce(p) {
  throw new Error(p);
}
function $t(p, t) {
  p || ce(t);
}
function uk(p) {
  switch (p == null ? void 0 : p.protocol) {
    case "http:":
    case "https:":
    case "ftp:":
    case "mailto:":
    case "tel:":
      return !0;
    default:
      return !1;
  }
}
function lA(p, t = null, e = null) {
  if (!p)
    return null;
  if (e && typeof p == "string") {
    if (e.addDefaultProtocol && p.startsWith("www.")) {
      const s = p.match(/\./g);
      (s == null ? void 0 : s.length) >= 2 && (p = `http://${p}`);
    }
    if (e.tryConvertEncoding)
      try {
        p = gk(p);
      } catch {
      }
  }
  const n = t ? URL.parse(p, t) : URL.parse(p);
  return uk(n) ? n : null;
}
function cA(p, t, e = !1) {
  const n = URL.parse(p);
  return n ? (n.hash = t, n.href) : e && lA(p, "http://example.com") ? p.split("#", 1)[0] + `${t ? `#${t}` : ""}` : "";
}
function It(p, t, e, n = !1) {
  return Object.defineProperty(p, t, {
    value: e,
    enumerable: !n,
    configurable: !0,
    writable: !1
  }), e;
}
const fc = function() {
  function t(e, n) {
    this.message = e, this.name = n;
  }
  return t.prototype = new Error(), t.constructor = t, t;
}();
class PE extends fc {
  constructor(t, e) {
    super(t, "PasswordException"), this.code = e;
  }
}
class C0 extends fc {
  constructor(t, e) {
    super(t, "UnknownErrorException"), this.details = e;
  }
}
class $m extends fc {
  constructor(t) {
    super(t, "InvalidPDFException");
  }
}
class Gd extends fc {
  constructor(t, e, n) {
    super(t, "ResponseException"), this.status = e, this.missing = n;
  }
}
class hk extends fc {
  constructor(t) {
    super(t, "FormatError");
  }
}
class ao extends fc {
  constructor(t) {
    super(t, "AbortException");
  }
}
function A1(p) {
  (typeof p != "object" || (p == null ? void 0 : p.length) === void 0) && ce("Invalid argument for bytesToString");
  const t = p.length, e = 8192;
  if (t < e)
    return String.fromCharCode.apply(null, p);
  const n = [];
  for (let s = 0; s < t; s += e) {
    const a = Math.min(s + e, t), l = p.subarray(s, a);
    n.push(String.fromCharCode.apply(null, l));
  }
  return n.join("");
}
function Fp(p) {
  typeof p != "string" && ce("Invalid argument for stringToBytes");
  const t = p.length, e = new Uint8Array(t);
  for (let n = 0; n < t; ++n)
    e[n] = p.charCodeAt(n) & 255;
  return e;
}
function dk(p) {
  return String.fromCharCode(p >> 24 & 255, p >> 16 & 255, p >> 8 & 255, p & 255);
}
function fk() {
  const p = new Uint8Array(4);
  return p[0] = 1, new Uint32Array(p.buffer, 0, 1)[0] === 1;
}
function pk() {
  try {
    return new Function(""), !0;
  } catch {
    return !1;
  }
}
class cn {
  static get isLittleEndian() {
    return It(this, "isLittleEndian", fk());
  }
  static get isEvalSupported() {
    return It(this, "isEvalSupported", pk());
  }
  static get isOffscreenCanvasSupported() {
    return It(this, "isOffscreenCanvasSupported", typeof OffscreenCanvas < "u");
  }
  static get isImageDecoderSupported() {
    return It(this, "isImageDecoderSupported", typeof ImageDecoder < "u");
  }
  static get platform() {
    const {
      platform: t,
      userAgent: e
    } = navigator;
    return It(this, "platform", {
      isAndroid: e.includes("Android"),
      isLinux: t.includes("Linux"),
      isMac: t.includes("Mac"),
      isWindows: t.includes("Win"),
      isFirefox: e.includes("Firefox")
    });
  }
  static get isCSSRoundSupported() {
    var t, e;
    return It(this, "isCSSRoundSupported", (e = (t = globalThis.CSS) == null ? void 0 : t.supports) == null ? void 0 : e.call(t, "width: round(1.5px, 1px)"));
  }
}
const x0 = Array.from(Array(256).keys(), (p) => p.toString(16).padStart(2, "0"));
var ro, hm, N0;
class ft {
  static makeHexColor(t, e, n) {
    return `#${x0[t]}${x0[e]}${x0[n]}`;
  }
  static domMatrixToTransform(t) {
    return [t.a, t.b, t.c, t.d, t.e, t.f];
  }
  static scaleMinMax(t, e) {
    let n;
    t[0] ? (t[0] < 0 && (n = e[0], e[0] = e[2], e[2] = n), e[0] *= t[0], e[2] *= t[0], t[3] < 0 && (n = e[1], e[1] = e[3], e[3] = n), e[1] *= t[3], e[3] *= t[3]) : (n = e[0], e[0] = e[1], e[1] = n, n = e[2], e[2] = e[3], e[3] = n, t[1] < 0 && (n = e[1], e[1] = e[3], e[3] = n), e[1] *= t[1], e[3] *= t[1], t[2] < 0 && (n = e[0], e[0] = e[2], e[2] = n), e[0] *= t[2], e[2] *= t[2]), e[0] += t[4], e[1] += t[5], e[2] += t[4], e[3] += t[5];
  }
  static transform(t, e) {
    return [t[0] * e[0] + t[2] * e[1], t[1] * e[0] + t[3] * e[1], t[0] * e[2] + t[2] * e[3], t[1] * e[2] + t[3] * e[3], t[0] * e[4] + t[2] * e[5] + t[4], t[1] * e[4] + t[3] * e[5] + t[5]];
  }
  static multiplyByDOMMatrix(t, e) {
    return [t[0] * e.a + t[2] * e.b, t[1] * e.a + t[3] * e.b, t[0] * e.c + t[2] * e.d, t[1] * e.c + t[3] * e.d, t[0] * e.e + t[2] * e.f + t[4], t[1] * e.e + t[3] * e.f + t[5]];
  }
  static applyTransform(t, e, n = 0) {
    const s = t[n], a = t[n + 1];
    t[n] = s * e[0] + a * e[2] + e[4], t[n + 1] = s * e[1] + a * e[3] + e[5];
  }
  static applyTransformToBezier(t, e, n = 0) {
    const s = e[0], a = e[1], l = e[2], u = e[3], h = e[4], f = e[5];
    for (let g = 0; g < 6; g += 2) {
      const y = t[n + g], v = t[n + g + 1];
      t[n + g] = y * s + v * l + h, t[n + g + 1] = y * a + v * u + f;
    }
  }
  static applyInverseTransform(t, e) {
    const n = t[0], s = t[1], a = e[0] * e[3] - e[1] * e[2];
    t[0] = (n * e[3] - s * e[2] + e[2] * e[5] - e[4] * e[3]) / a, t[1] = (-n * e[1] + s * e[0] + e[4] * e[1] - e[5] * e[0]) / a;
  }
  static axialAlignedBoundingBox(t, e, n) {
    const s = e[0], a = e[1], l = e[2], u = e[3], h = e[4], f = e[5], g = t[0], y = t[1], v = t[2], S = t[3];
    let x = s * g + h, _ = x, k = s * v + h, T = k, R = u * y + f, P = R, L = u * S + f, D = L;
    if (a !== 0 || l !== 0) {
      const N = a * g, B = a * v, $ = l * y, V = l * S;
      x += $, T += $, k += V, _ += V, R += N, D += N, L += B, P += B;
    }
    n[0] = Math.min(n[0], x, k, _, T), n[1] = Math.min(n[1], R, L, P, D), n[2] = Math.max(n[2], x, k, _, T), n[3] = Math.max(n[3], R, L, P, D);
  }
  static inverseTransform(t) {
    const e = t[0] * t[3] - t[1] * t[2];
    return [t[3] / e, -t[1] / e, -t[2] / e, t[0] / e, (t[2] * t[5] - t[4] * t[3]) / e, (t[4] * t[1] - t[5] * t[0]) / e];
  }
  static singularValueDecompose2dScale(t, e) {
    const n = t[0], s = t[1], a = t[2], l = t[3], u = n ** 2 + s ** 2, h = n * a + s * l, f = a ** 2 + l ** 2, g = (u + f) / 2, y = Math.sqrt(g ** 2 - (u * f - h ** 2));
    e[0] = Math.sqrt(g + y || 1), e[1] = Math.sqrt(g - y || 1);
  }
  static normalizeRect(t) {
    const e = t.slice(0);
    return t[0] > t[2] && (e[0] = t[2], e[2] = t[0]), t[1] > t[3] && (e[1] = t[3], e[3] = t[1]), e;
  }
  static intersect(t, e) {
    const n = Math.max(Math.min(t[0], t[2]), Math.min(e[0], e[2])), s = Math.min(Math.max(t[0], t[2]), Math.max(e[0], e[2]));
    if (n > s)
      return null;
    const a = Math.max(Math.min(t[1], t[3]), Math.min(e[1], e[3])), l = Math.min(Math.max(t[1], t[3]), Math.max(e[1], e[3]));
    return a > l ? null : [n, a, s, l];
  }
  static pointBoundingBox(t, e, n) {
    n[0] = Math.min(n[0], t), n[1] = Math.min(n[1], e), n[2] = Math.max(n[2], t), n[3] = Math.max(n[3], e);
  }
  static rectBoundingBox(t, e, n, s, a) {
    a[0] = Math.min(a[0], t, n), a[1] = Math.min(a[1], e, s), a[2] = Math.max(a[2], t, n), a[3] = Math.max(a[3], e, s);
  }
  static bezierBoundingBox(t, e, n, s, a, l, u, h, f) {
    f[0] = Math.min(f[0], t, u), f[1] = Math.min(f[1], e, h), f[2] = Math.max(f[2], t, u), f[3] = Math.max(f[3], e, h), E(this, ro, N0).call(this, t, n, a, u, e, s, l, h, 3 * (-t + 3 * (n - a) + u), 6 * (t - 2 * n + a), 3 * (n - t), f), E(this, ro, N0).call(this, t, n, a, u, e, s, l, h, 3 * (-e + 3 * (s - l) + h), 6 * (e - 2 * s + l), 3 * (s - e), f);
  }
}
ro = new WeakSet(), hm = function(t, e, n, s, a, l, u, h, f, g) {
  if (f <= 0 || f >= 1)
    return;
  const y = 1 - f, v = f * f, S = v * f, x = y * (y * (y * t + 3 * f * e) + 3 * v * n) + S * s, _ = y * (y * (y * a + 3 * f * l) + 3 * v * u) + S * h;
  g[0] = Math.min(g[0], x), g[1] = Math.min(g[1], _), g[2] = Math.max(g[2], x), g[3] = Math.max(g[3], _);
}, N0 = function(t, e, n, s, a, l, u, h, f, g, y, v) {
  if (Math.abs(f) < 1e-12) {
    Math.abs(g) >= 1e-12 && E(this, ro, hm).call(this, t, e, n, s, a, l, u, h, -y / g, v);
    return;
  }
  const S = g ** 2 - 4 * y * f;
  if (S < 0)
    return;
  const x = Math.sqrt(S), _ = 2 * f;
  E(this, ro, hm).call(this, t, e, n, s, a, l, u, h, (-g + x) / _, v), E(this, ro, hm).call(this, t, e, n, s, a, l, u, h, (-g - x) / _, v);
}, b(ft, ro);
function gk(p) {
  return decodeURIComponent(escape(p));
}
let T0 = null, RE = null;
function S1(p) {
  return T0 || (T0 = /([\u00a0\u00b5\u037e\u0eb3\u2000-\u200a\u202f\u2126\ufb00-\ufb04\ufb06\ufb20-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufba1\ufba4-\ufba9\ufbae-\ufbb1\ufbd3-\ufbdc\ufbde-\ufbe7\ufbea-\ufbf8\ufbfc-\ufbfd\ufc00-\ufc5d\ufc64-\ufcf1\ufcf5-\ufd3d\ufd88\ufdf4\ufdfa-\ufdfb\ufe71\ufe77\ufe79\ufe7b\ufe7d]+)|(\ufb05+)/gu, RE = /* @__PURE__ */ new Map([["ﬅ", "ſt"]])), p.replaceAll(T0, (t, e, n) => e ? e.normalize("NFKC") : RE.get(n));
}
function uA() {
  if (typeof crypto.randomUUID == "function")
    return crypto.randomUUID();
  const p = new Uint8Array(32);
  return crypto.getRandomValues(p), A1(p);
}
const hA = "pdfjs_internal_id_";
function mk(p, t, e) {
  if (!Array.isArray(e) || e.length < 2)
    return !1;
  const [n, s, ...a] = e;
  if (!p(n) && !Number.isInteger(n) || !t(s))
    return !1;
  const l = a.length;
  let u = !0;
  switch (s.name) {
    case "XYZ":
      if (l < 2 || l > 3)
        return !1;
      break;
    case "Fit":
    case "FitB":
      return l === 0;
    case "FitH":
    case "FitBH":
    case "FitV":
    case "FitBV":
      if (l > 1)
        return !1;
      break;
    case "FitR":
      if (l !== 4)
        return !1;
      u = !1;
      break;
    default:
      return !1;
  }
  for (const h of a)
    if (!(typeof h == "number" || u && h === null))
      return !1;
  return !0;
}
function Ln(p, t, e) {
  return Math.min(Math.max(p, t), e);
}
function b1(p) {
  return Uint8Array.prototype.toBase64 ? p.toBase64() : btoa(A1(p));
}
function yk(p) {
  return Uint8Array.fromBase64 ? Uint8Array.fromBase64(p) : Fp(atob(p));
}
typeof Promise.try != "function" && (Promise.try = function(p, ...t) {
  return new Promise((e) => {
    e(p(...t));
  });
});
typeof Math.sumPrecise != "function" && (Math.sumPrecise = function(p) {
  return p.reduce((t, e) => t + e, 0);
});
class Xd {
  static textContent(t) {
    const e = [], n = {
      items: e,
      styles: /* @__PURE__ */ Object.create(null)
    };
    function s(a) {
      var h;
      if (!a)
        return;
      let l = null;
      const u = a.name;
      if (u === "#text")
        l = a.value;
      else if (Xd.shouldBuildText(u))
        (h = a == null ? void 0 : a.attributes) != null && h.textContent ? l = a.attributes.textContent : a.value && (l = a.value);
      else return;
      if (l !== null && e.push({
        str: l
      }), !!a.children)
        for (const f of a.children)
          s(f);
    }
    return s(t), n;
  }
  static shouldBuildText(t) {
    return !(t === "textarea" || t === "input" || t === "option" || t === "select");
  }
}
class dA {
  static setupStorage(t, e, n, s, a) {
    const l = s.getValue(e, {
      value: null
    });
    switch (n.name) {
      case "textarea":
        if (l.value !== null && (t.textContent = l.value), a === "print")
          break;
        t.addEventListener("input", (u) => {
          s.setValue(e, {
            value: u.target.value
          });
        });
        break;
      case "input":
        if (n.attributes.type === "radio" || n.attributes.type === "checkbox") {
          if (l.value === n.attributes.xfaOn ? t.setAttribute("checked", !0) : l.value === n.attributes.xfaOff && t.removeAttribute("checked"), a === "print")
            break;
          t.addEventListener("change", (u) => {
            s.setValue(e, {
              value: u.target.checked ? u.target.getAttribute("xfaOn") : u.target.getAttribute("xfaOff")
            });
          });
        } else {
          if (l.value !== null && t.setAttribute("value", l.value), a === "print")
            break;
          t.addEventListener("input", (u) => {
            s.setValue(e, {
              value: u.target.value
            });
          });
        }
        break;
      case "select":
        if (l.value !== null) {
          t.setAttribute("value", l.value);
          for (const u of n.children)
            u.attributes.value === l.value ? u.attributes.selected = !0 : u.attributes.hasOwnProperty("selected") && delete u.attributes.selected;
        }
        t.addEventListener("input", (u) => {
          const h = u.target.options, f = h.selectedIndex === -1 ? "" : h[h.selectedIndex].value;
          s.setValue(e, {
            value: f
          });
        });
        break;
    }
  }
  static setAttributes({
    html: t,
    element: e,
    storage: n = null,
    intent: s,
    linkService: a
  }) {
    const {
      attributes: l
    } = e, u = t instanceof HTMLAnchorElement;
    l.type === "radio" && (l.name = `${l.name}-${s}`);
    for (const [h, f] of Object.entries(l))
      if (f != null)
        switch (h) {
          case "class":
            f.length && t.setAttribute(h, f.join(" "));
            break;
          case "dataId":
            break;
          case "id":
            t.setAttribute("data-element-id", f);
            break;
          case "style":
            Object.assign(t.style, f);
            break;
          case "textContent":
            t.textContent = f;
            break;
          default:
            (!u || h !== "href" && h !== "newWindow") && t.setAttribute(h, f);
        }
    u && a.addLinkAttributes(t, l.href, l.newWindow), n && l.dataId && this.setupStorage(t, l.dataId, e, n);
  }
  static render(t) {
    var y, v;
    const e = t.annotationStorage, n = t.linkService, s = t.xfaHtml, a = t.intent || "display", l = document.createElement(s.name);
    s.attributes && this.setAttributes({
      html: l,
      element: s,
      intent: a,
      linkService: n
    });
    const u = a !== "richText", h = t.div;
    if (h.append(l), t.viewport) {
      const S = `matrix(${t.viewport.transform.join(",")})`;
      h.style.transform = S;
    }
    u && h.setAttribute("class", "xfaLayer xfaFont");
    const f = [];
    if (s.children.length === 0) {
      if (s.value) {
        const S = document.createTextNode(s.value);
        l.append(S), u && Xd.shouldBuildText(s.name) && f.push(S);
      }
      return {
        textDivs: f
      };
    }
    const g = [[s, -1, l]];
    for (; g.length > 0; ) {
      const [S, x, _] = g.at(-1);
      if (x + 1 === S.children.length) {
        g.pop();
        continue;
      }
      const k = S.children[++g.at(-1)[1]];
      if (k === null)
        continue;
      const {
        name: T
      } = k;
      if (T === "#text") {
        const P = document.createTextNode(k.value);
        f.push(P), _.append(P);
        continue;
      }
      const R = (y = k == null ? void 0 : k.attributes) != null && y.xmlns ? document.createElementNS(k.attributes.xmlns, T) : document.createElement(T);
      if (_.append(R), k.attributes && this.setAttributes({
        html: R,
        element: k,
        storage: e,
        intent: a,
        linkService: n
      }), ((v = k.children) == null ? void 0 : v.length) > 0)
        g.push([k, -1, R]);
      else if (k.value) {
        const P = document.createTextNode(k.value);
        u && Xd.shouldBuildText(T) && f.push(P), R.append(P);
      }
    }
    for (const S of h.querySelectorAll(".xfaNonInteractive input, .xfaNonInteractive textarea"))
      S.setAttribute("readOnly", !0);
    return {
      textDivs: f
    };
  }
  static update(t) {
    const e = `matrix(${t.viewport.transform.join(",")})`;
    t.div.style.transform = e, t.div.hidden = !1;
  }
}
const Ar = "http://www.w3.org/2000/svg", Ka = class Ka {
};
Z(Ka, "CSS", 96), Z(Ka, "PDF", 72), Z(Ka, "PDF_TO_CSS_UNITS", Ka.CSS / Ka.PDF);
let lo = Ka;
async function Lh(p, t = "text") {
  if (Ed(p, document.baseURI)) {
    const e = await fetch(p);
    if (!e.ok)
      throw new Error(e.statusText);
    switch (t) {
      case "arraybuffer":
        return e.arrayBuffer();
      case "blob":
        return e.blob();
      case "json":
        return e.json();
    }
    return e.text();
  }
  return new Promise((e, n) => {
    const s = new XMLHttpRequest();
    s.open("GET", p, !0), s.responseType = t, s.onreadystatechange = () => {
      if (s.readyState === XMLHttpRequest.DONE) {
        if (s.status === 200 || s.status === 0) {
          switch (t) {
            case "arraybuffer":
            case "blob":
            case "json":
              e(s.response);
              return;
          }
          e(s.responseText);
          return;
        }
        n(new Error(s.statusText));
      }
    }, s.send(null);
  });
}
class Np {
  constructor({
    viewBox: t,
    userUnit: e,
    scale: n,
    rotation: s,
    offsetX: a = 0,
    offsetY: l = 0,
    dontFlip: u = !1
  }) {
    this.viewBox = t, this.userUnit = e, this.scale = n, this.rotation = s, this.offsetX = a, this.offsetY = l, n *= e;
    const h = (t[2] + t[0]) / 2, f = (t[3] + t[1]) / 2;
    let g, y, v, S;
    switch (s %= 360, s < 0 && (s += 360), s) {
      case 180:
        g = -1, y = 0, v = 0, S = 1;
        break;
      case 90:
        g = 0, y = 1, v = 1, S = 0;
        break;
      case 270:
        g = 0, y = -1, v = -1, S = 0;
        break;
      case 0:
        g = 1, y = 0, v = 0, S = -1;
        break;
      default:
        throw new Error("PageViewport: Invalid rotation, must be a multiple of 90 degrees.");
    }
    u && (v = -v, S = -S);
    let x, _, k, T;
    g === 0 ? (x = Math.abs(f - t[1]) * n + a, _ = Math.abs(h - t[0]) * n + l, k = (t[3] - t[1]) * n, T = (t[2] - t[0]) * n) : (x = Math.abs(h - t[0]) * n + a, _ = Math.abs(f - t[1]) * n + l, k = (t[2] - t[0]) * n, T = (t[3] - t[1]) * n), this.transform = [g * n, y * n, v * n, S * n, x - g * n * h - v * n * f, _ - y * n * h - S * n * f], this.width = k, this.height = T;
  }
  get rawDims() {
    const t = this.viewBox;
    return It(this, "rawDims", {
      pageWidth: t[2] - t[0],
      pageHeight: t[3] - t[1],
      pageX: t[0],
      pageY: t[1]
    });
  }
  clone({
    scale: t = this.scale,
    rotation: e = this.rotation,
    offsetX: n = this.offsetX,
    offsetY: s = this.offsetY,
    dontFlip: a = !1
  } = {}) {
    return new Np({
      viewBox: this.viewBox.slice(),
      userUnit: this.userUnit,
      scale: t,
      rotation: e,
      offsetX: n,
      offsetY: s,
      dontFlip: a
    });
  }
  convertToViewportPoint(t, e) {
    const n = [t, e];
    return ft.applyTransform(n, this.transform), n;
  }
  convertToViewportRectangle(t) {
    const e = [t[0], t[1]];
    ft.applyTransform(e, this.transform);
    const n = [t[2], t[3]];
    return ft.applyTransform(n, this.transform), [e[0], e[1], n[0], n[1]];
  }
  convertToPdfPoint(t, e) {
    const n = [t, e];
    return ft.applyInverseTransform(n, this.transform), n;
  }
}
class by extends fc {
  constructor(t, e = 0) {
    super(t, "RenderingCancelledException"), this.extraDelay = e;
  }
}
function Op(p) {
  const t = p.length;
  let e = 0;
  for (; e < t && p[e].trim() === ""; )
    e++;
  return p.substring(e, e + 5).toLowerCase() === "data:";
}
function Ey(p) {
  return typeof p == "string" && /\.pdf$/i.test(p);
}
function E1(p) {
  return [p] = p.split(/[#?]/, 1), p.substring(p.lastIndexOf("/") + 1);
}
function _1(p, t = "document.pdf") {
  if (typeof p != "string")
    return t;
  if (Op(p))
    return Rt('getPdfFilenameFromUrl: ignore "data:"-URL for performance reasons.'), t;
  const n = ((u) => {
    try {
      return new URL(u);
    } catch {
      try {
        return new URL(decodeURIComponent(u));
      } catch {
        try {
          return new URL(u, "https://foo.bar");
        } catch {
          try {
            return new URL(decodeURIComponent(u), "https://foo.bar");
          } catch {
            return null;
          }
        }
      }
    }
  })(p);
  if (!n)
    return t;
  const s = (u) => {
    try {
      let h = decodeURIComponent(u);
      return h.includes("/") ? (h = h.split("/").at(-1), h.test(/^\.pdf$/i) ? h : u) : h;
    } catch {
      return u;
    }
  }, a = /\.pdf$/i, l = n.pathname.split("/").at(-1);
  if (a.test(l))
    return s(l);
  if (n.searchParams.size > 0) {
    const u = Array.from(n.searchParams.values()).reverse();
    for (const f of u)
      if (a.test(f))
        return s(f);
    const h = Array.from(n.searchParams.keys()).reverse();
    for (const f of h)
      if (a.test(f))
        return s(f);
  }
  if (n.hash) {
    const h = /[^/?#=]+\.pdf\b(?!.*\.pdf\b)/i.exec(n.hash);
    if (h)
      return s(h[0]);
  }
  return t;
}
class ME {
  constructor() {
    Z(this, "started", /* @__PURE__ */ Object.create(null));
    Z(this, "times", []);
  }
  time(t) {
    t in this.started && Rt(`Timer is already running for ${t}`), this.started[t] = Date.now();
  }
  timeEnd(t) {
    t in this.started || Rt(`Timer has not been started for ${t}`), this.times.push({
      name: t,
      start: this.started[t],
      end: Date.now()
    }), delete this.started[t];
  }
  toString() {
    const t = [];
    let e = 0;
    for (const {
      name: n
    } of this.times)
      e = Math.max(n.length, e);
    for (const {
      name: n,
      start: s,
      end: a
    } of this.times)
      t.push(`${n.padEnd(e)} ${a - s}ms
`);
    return t.join("");
  }
}
function Ed(p, t) {
  const e = t ? URL.parse(p, t) : URL.parse(p);
  return (e == null ? void 0 : e.protocol) === "http:" || (e == null ? void 0 : e.protocol) === "https:";
}
function Li(p) {
  p.preventDefault();
}
function Ce(p) {
  p.preventDefault(), p.stopPropagation();
}
function vk(p) {
  console.log("Deprecated API usage: " + p);
}
var Jd;
class qd {
  static toDateObject(t) {
    if (t instanceof Date)
      return t;
    if (!t || typeof t != "string")
      return null;
    o(this, Jd) || w(this, Jd, new RegExp("^D:(\\d{4})(\\d{2})?(\\d{2})?(\\d{2})?(\\d{2})?(\\d{2})?([Z|+|-])?(\\d{2})?'?(\\d{2})?'?"));
    const e = o(this, Jd).exec(t);
    if (!e)
      return null;
    const n = parseInt(e[1], 10);
    let s = parseInt(e[2], 10);
    s = s >= 1 && s <= 12 ? s - 1 : 0;
    let a = parseInt(e[3], 10);
    a = a >= 1 && a <= 31 ? a : 1;
    let l = parseInt(e[4], 10);
    l = l >= 0 && l <= 23 ? l : 0;
    let u = parseInt(e[5], 10);
    u = u >= 0 && u <= 59 ? u : 0;
    let h = parseInt(e[6], 10);
    h = h >= 0 && h <= 59 ? h : 0;
    const f = e[7] || "Z";
    let g = parseInt(e[8], 10);
    g = g >= 0 && g <= 23 ? g : 0;
    let y = parseInt(e[9], 10) || 0;
    return y = y >= 0 && y <= 59 ? y : 0, f === "-" ? (l += g, u += y) : f === "+" && (l -= g, u -= y), new Date(Date.UTC(n, s, a, l, u, h));
  }
}
Jd = new WeakMap(), b(qd, Jd);
function C1(p, {
  scale: t = 1,
  rotation: e = 0
}) {
  const {
    width: n,
    height: s
  } = p.attributes.style, a = [0, 0, parseInt(n), parseInt(s)];
  return new Np({
    viewBox: a,
    userUnit: 1,
    scale: t,
    rotation: e
  });
}
function Dh(p) {
  if (p.startsWith("#")) {
    const t = parseInt(p.slice(1), 16);
    return [(t & 16711680) >> 16, (t & 65280) >> 8, t & 255];
  }
  return p.startsWith("rgb(") ? p.slice(4, -1).split(",").map((t) => parseInt(t)) : p.startsWith("rgba(") ? p.slice(5, -1).split(",").map((t) => parseInt(t)).slice(0, 3) : (Rt(`Not a valid color format: "${p}"`), [0, 0, 0]);
}
function wk(p) {
  const t = document.createElement("span");
  t.style.visibility = "hidden", t.style.colorScheme = "only light", document.body.append(t);
  for (const e of p.keys()) {
    t.style.color = e;
    const n = window.getComputedStyle(t).color;
    p.set(e, Dh(n));
  }
  t.remove();
}
function _e(p) {
  const {
    a: t,
    b: e,
    c: n,
    d: s,
    e: a,
    f: l
  } = p.getTransform();
  return [t, e, n, s, a, l];
}
function Ns(p) {
  const {
    a: t,
    b: e,
    c: n,
    d: s,
    e: a,
    f: l
  } = p.getTransform().invertSelf();
  return [t, e, n, s, a, l];
}
function Ta(p, t, e = !1, n = !0) {
  if (t instanceof Np) {
    const {
      pageWidth: s,
      pageHeight: a
    } = t.rawDims, {
      style: l
    } = p, u = cn.isCSSRoundSupported, h = `var(--total-scale-factor) * ${s}px`, f = `var(--total-scale-factor) * ${a}px`, g = u ? `round(down, ${h}, var(--scale-round-x))` : `calc(${h})`, y = u ? `round(down, ${f}, var(--scale-round-y))` : `calc(${f})`;
    !e || t.rotation % 180 === 0 ? (l.width = g, l.height = y) : (l.width = y, l.height = g);
  }
  n && p.setAttribute("data-main-rotation", t.rotation);
}
class ks {
  constructor() {
    const {
      pixelRatio: t
    } = ks;
    this.sx = t, this.sy = t;
  }
  get scaled() {
    return this.sx !== 1 || this.sy !== 1;
  }
  get symmetric() {
    return this.sx === this.sy;
  }
  limitCanvas(t, e, n, s, a = -1) {
    let l = 1 / 0, u = 1 / 0, h = 1 / 0;
    n = ks.capPixels(n, a), n > 0 && (l = Math.sqrt(n / (t * e))), s !== -1 && (u = s / t, h = s / e);
    const f = Math.min(l, u, h);
    return this.sx > f || this.sy > f ? (this.sx = f, this.sy = f, !0) : !1;
  }
  static get pixelRatio() {
    return globalThis.devicePixelRatio || 1;
  }
  static capPixels(t, e) {
    if (e >= 0) {
      const n = Math.ceil(window.screen.availWidth * window.screen.availHeight * this.pixelRatio ** 2 * (1 + e / 100));
      return t > 0 ? Math.min(t, n) : n;
    }
    return t;
  }
}
const Vm = ["image/apng", "image/avif", "image/bmp", "image/gif", "image/jpeg", "image/png", "image/svg+xml", "image/webp", "image/x-icon"];
class Ak {
  static get isDarkMode() {
    var t;
    return It(this, "isDarkMode", !!((t = window == null ? void 0 : window.matchMedia) != null && t.call(window, "(prefers-color-scheme: dark)").matches));
  }
}
class x1 {
  static get commentForegroundColor() {
    const t = document.createElement("span");
    t.classList.add("comment", "sidebar");
    const {
      style: e
    } = t;
    e.width = e.height = "0", e.display = "none", e.color = "var(--comment-fg-color)", document.body.append(t);
    const {
      color: n
    } = window.getComputedStyle(t);
    return t.remove(), It(this, "commentForegroundColor", Dh(n));
  }
}
function T1(p, t, e, n) {
  n = Math.min(Math.max(n ?? 1, 0), 1);
  const s = 255 * (1 - n);
  return p = Math.round(p * n + s), t = Math.round(t * n + s), e = Math.round(e * n + s), [p, t, e];
}
function LE(p, t) {
  const e = p[0] / 255, n = p[1] / 255, s = p[2] / 255, a = Math.max(e, n, s), l = Math.min(e, n, s), u = (a + l) / 2;
  if (a === l)
    t[0] = t[1] = 0;
  else {
    const h = a - l;
    switch (t[1] = u < 0.5 ? h / (a + l) : h / (2 - a - l), a) {
      case e:
        t[0] = ((n - s) / h + (n < s ? 6 : 0)) * 60;
        break;
      case n:
        t[0] = ((s - e) / h + 2) * 60;
        break;
      case s:
        t[0] = ((e - n) / h + 4) * 60;
        break;
    }
  }
  t[2] = u;
}
function O0(p, t) {
  const e = p[0], n = p[1], s = p[2], a = (1 - Math.abs(2 * s - 1)) * n, l = a * (1 - Math.abs(e / 60 % 2 - 1)), u = s - a / 2;
  switch (Math.floor(e / 60)) {
    case 0:
      t[0] = a + u, t[1] = l + u, t[2] = u;
      break;
    case 1:
      t[0] = l + u, t[1] = a + u, t[2] = u;
      break;
    case 2:
      t[0] = u, t[1] = a + u, t[2] = l + u;
      break;
    case 3:
      t[0] = u, t[1] = l + u, t[2] = a + u;
      break;
    case 4:
      t[0] = l + u, t[1] = u, t[2] = a + u;
      break;
    case 5:
    case 6:
      t[0] = a + u, t[1] = u, t[2] = l + u;
      break;
  }
}
function DE(p) {
  return p <= 0.03928 ? p / 12.92 : ((p + 0.055) / 1.055) ** 2.4;
}
function IE(p, t, e) {
  O0(p, e), e.map(DE);
  const n = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
  O0(t, e), e.map(DE);
  const s = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
  return n > s ? (n + 0.05) / (s + 0.05) : (s + 0.05) / (n + 0.05);
}
const FE = /* @__PURE__ */ new Map();
function k1(p, t) {
  const e = p[0] + p[1] * 256 + p[2] * 65536 + t[0] * 16777216 + t[1] * 4294967296 + t[2] * 1099511627776;
  let n = FE.get(e);
  if (n)
    return n;
  const s = new Float32Array(9), a = s.subarray(0, 3), l = s.subarray(3, 6);
  LE(p, l);
  const u = s.subarray(6, 9);
  LE(t, u);
  const h = u[2] < 0.5, f = h ? 12 : 4.5;
  if (l[2] = h ? Math.sqrt(l[2]) : 1 - Math.sqrt(1 - l[2]), IE(l, u, a) < f) {
    let g, y;
    h ? (g = l[2], y = 1) : (g = 0, y = l[2]);
    const v = 5e-3;
    for (; y - g > v; ) {
      const S = l[2] = (g + y) / 2;
      h === IE(l, u, a) < f ? g = S : y = S;
    }
    l[2] = h ? y : g;
  }
  return O0(l, a), n = ft.makeHexColor(Math.round(a[0] * 255), Math.round(a[1] * 255), Math.round(a[2] * 255)), FE.set(e, n), n;
}
function fA({
  html: p,
  dir: t,
  className: e
}, n) {
  const s = document.createDocumentFragment();
  if (typeof p == "string") {
    const a = document.createElement("p");
    a.dir = t || "auto";
    const l = p.split(/(?:\r\n?|\n)/);
    for (let u = 0, h = l.length; u < h; ++u) {
      const f = l[u];
      a.append(document.createTextNode(f)), u < h - 1 && a.append(document.createElement("br"));
    }
    s.append(a);
  } else
    dA.render({
      xfaHtml: p,
      div: s,
      intent: "richText"
    });
  s.firstChild.classList.add("richText", e), n.append(s);
}
var Do, Io, $i, Vi, tf, Fo, iu, su, ef, ey, P1, An, R1, M1, jc, _d;
const Er = class Er {
  constructor(t) {
    b(this, An);
    b(this, Do, null);
    b(this, Io, null);
    b(this, $i);
    b(this, Vi, null);
    b(this, tf, null);
    b(this, Fo, null);
    b(this, iu, null);
    b(this, su, null);
    w(this, $i, t), o(Er, ef) || w(Er, ef, Object.freeze({
      freetext: "pdfjs-editor-remove-freetext-button",
      highlight: "pdfjs-editor-remove-highlight-button",
      ink: "pdfjs-editor-remove-ink-button",
      stamp: "pdfjs-editor-remove-stamp-button",
      signature: "pdfjs-editor-remove-signature-button"
    }));
  }
  render() {
    const t = w(this, Do, document.createElement("div"));
    t.classList.add("editToolbar", "hidden"), t.setAttribute("role", "toolbar");
    const e = o(this, $i)._uiManager._signal;
    e instanceof AbortSignal && !e.aborted && (t.addEventListener("contextmenu", Li, {
      signal: e
    }), t.addEventListener("pointerdown", E(Er, ey, P1), {
      signal: e
    }));
    const n = w(this, Vi, document.createElement("div"));
    n.className = "buttons", t.append(n);
    const s = o(this, $i).toolbarPosition;
    if (s) {
      const {
        style: a
      } = t, l = o(this, $i)._uiManager.direction === "ltr" ? 1 - s[0] : s[0];
      a.insetInlineEnd = `${100 * l}%`, a.top = `calc(${100 * s[1]}% + var(--editor-toolbar-vert-offset))`;
    }
    return t;
  }
  get div() {
    return o(this, Do);
  }
  hide() {
    var t;
    o(this, Do).classList.add("hidden"), (t = o(this, Io)) == null || t.hideDropdown();
  }
  show() {
    var t, e;
    o(this, Do).classList.remove("hidden"), (t = o(this, tf)) == null || t.shown(), (e = o(this, Fo)) == null || e.shown();
  }
  addDeleteButton() {
    const {
      editorType: t,
      _uiManager: e
    } = o(this, $i), n = document.createElement("button");
    n.classList.add("basic", "deleteButton"), n.tabIndex = 0, n.setAttribute("data-l10n-id", o(Er, ef)[t]), E(this, An, jc).call(this, n) && n.addEventListener("click", (s) => {
      e.delete();
    }, {
      signal: e._signal
    }), o(this, Vi).append(n);
  }
  async addAltText(t) {
    const e = await t.render();
    E(this, An, jc).call(this, e), o(this, Vi).append(e, o(this, An, _d)), w(this, tf, t);
  }
  addComment(t, e = null) {
    if (o(this, Fo))
      return;
    const n = t.renderForToolbar();
    if (!n)
      return;
    E(this, An, jc).call(this, n);
    const s = w(this, iu, o(this, An, _d));
    e ? (o(this, Vi).insertBefore(n, e), o(this, Vi).insertBefore(s, e)) : o(this, Vi).append(n, s), w(this, Fo, t), t.toolbar = this;
  }
  addColorPicker(t) {
    if (o(this, Io))
      return;
    w(this, Io, t);
    const e = t.renderButton();
    E(this, An, jc).call(this, e), o(this, Vi).append(e, o(this, An, _d));
  }
  async addEditSignatureButton(t) {
    const e = w(this, su, await t.renderEditButton(o(this, $i)));
    E(this, An, jc).call(this, e), o(this, Vi).append(e, o(this, An, _d));
  }
  removeButton(t) {
    var e, n;
    switch (t) {
      case "comment":
        (e = o(this, Fo)) == null || e.removeToolbarCommentButton(), w(this, Fo, null), (n = o(this, iu)) == null || n.remove(), w(this, iu, null);
        break;
    }
  }
  async addButton(t, e) {
    switch (t) {
      case "colorPicker":
        this.addColorPicker(e);
        break;
      case "altText":
        await this.addAltText(e);
        break;
      case "editSignature":
        await this.addEditSignatureButton(e);
        break;
      case "delete":
        this.addDeleteButton();
        break;
      case "comment":
        this.addComment(e);
        break;
    }
  }
  async addButtonBefore(t, e, n) {
    const s = o(this, Vi).querySelector(n);
    s && t === "comment" && this.addComment(e, s);
  }
  updateEditSignatureButton(t) {
    o(this, su) && (o(this, su).title = t);
  }
  remove() {
    var t;
    o(this, Do).remove(), (t = o(this, Io)) == null || t.destroy(), w(this, Io, null);
  }
};
Do = new WeakMap(), Io = new WeakMap(), $i = new WeakMap(), Vi = new WeakMap(), tf = new WeakMap(), Fo = new WeakMap(), iu = new WeakMap(), su = new WeakMap(), ef = new WeakMap(), ey = new WeakSet(), P1 = function(t) {
  t.stopPropagation();
}, An = new WeakSet(), R1 = function(t) {
  o(this, $i)._focusEventsAllowed = !1, Ce(t);
}, M1 = function(t) {
  o(this, $i)._focusEventsAllowed = !0, Ce(t);
}, jc = function(t) {
  const e = o(this, $i)._uiManager._signal;
  return !(e instanceof AbortSignal) || e.aborted ? !1 : (t.addEventListener("focusin", E(this, An, R1).bind(this), {
    capture: !0,
    signal: e
  }), t.addEventListener("focusout", E(this, An, M1).bind(this), {
    capture: !0,
    signal: e
  }), t.addEventListener("contextmenu", Li, {
    signal: e
  }), !0);
}, _d = function() {
  const t = document.createElement("div");
  return t.className = "divider", t;
}, b(Er, ey), b(Er, ef, null);
let B0 = Er;
var nf, Ja, Cr, co, L1, D1, z0;
class Sk {
  constructor(t) {
    b(this, co);
    b(this, nf, null);
    b(this, Ja, null);
    b(this, Cr);
    w(this, Cr, t);
  }
  show(t, e, n) {
    const [s, a] = E(this, co, D1).call(this, e, n), {
      style: l
    } = o(this, Ja) || w(this, Ja, E(this, co, L1).call(this));
    t.append(o(this, Ja)), l.insetInlineEnd = `${100 * s}%`, l.top = `calc(${100 * a}% + var(--editor-toolbar-vert-offset))`;
  }
  hide() {
    o(this, Ja).remove();
  }
}
nf = new WeakMap(), Ja = new WeakMap(), Cr = new WeakMap(), co = new WeakSet(), L1 = function() {
  const t = w(this, Ja, document.createElement("div"));
  t.className = "editToolbar", t.setAttribute("role", "toolbar");
  const e = o(this, Cr)._signal;
  e instanceof AbortSignal && !e.aborted && t.addEventListener("contextmenu", Li, {
    signal: e
  });
  const n = w(this, nf, document.createElement("div"));
  return n.className = "buttons", t.append(n), o(this, Cr).hasCommentManager() && E(this, co, z0).call(this, "commentButton", "pdfjs-comment-floating-button", "pdfjs-comment-floating-button-label", () => {
    o(this, Cr).commentSelection("floating_button");
  }), E(this, co, z0).call(this, "highlightButton", "pdfjs-highlight-floating-button1", "pdfjs-highlight-floating-button-label", () => {
    o(this, Cr).highlightSelection("floating_button");
  }), t;
}, D1 = function(t, e) {
  let n = 0, s = 0;
  for (const a of t) {
    const l = a.y + a.height;
    if (l < n)
      continue;
    const u = a.x + (e ? a.width : 0);
    if (l > n) {
      s = u, n = l;
      continue;
    }
    e ? u > s && (s = u) : u < s && (s = u);
  }
  return [e ? 1 - s : s, n];
}, z0 = function(t, e, n, s) {
  const a = document.createElement("button");
  a.classList.add("basic", t), a.tabIndex = 0, a.setAttribute("data-l10n-id", e);
  const l = document.createElement("span");
  a.append(l), l.className = "visuallyHidden", l.setAttribute("data-l10n-id", n);
  const u = o(this, Cr)._signal;
  u instanceof AbortSignal && !u.aborted && (a.addEventListener("contextmenu", Li, {
    signal: u
  }), a.addEventListener("click", s, {
    signal: u
  })), o(this, nf).append(a);
};
function I1(p, t, e) {
  for (const n of e)
    t.addEventListener(n, p[n].bind(p));
}
var ny;
class bk {
  constructor() {
    b(this, ny, 0);
  }
  get id() {
    return `${Wd}${Ge(this, ny)._++}`;
  }
}
ny = new WeakMap();
var ru, sf, xn, ou, dm;
const _A = class _A {
  constructor() {
    b(this, ou);
    b(this, ru, uA());
    b(this, sf, 0);
    b(this, xn, null);
  }
  static get _isSVGFittingCanvas() {
    const t = 'data:image/svg+xml;charset=UTF-8,<svg viewBox="0 0 1 1" width="1" height="1" xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1" style="fill:red;"/></svg>', n = new OffscreenCanvas(1, 3).getContext("2d", {
      willReadFrequently: !0
    }), s = new Image();
    s.src = t;
    const a = s.decode().then(() => (n.drawImage(s, 0, 0, 1, 1, 0, 0, 1, 3), new Uint32Array(n.getImageData(0, 0, 1, 1).data.buffer)[0] === 0));
    return It(this, "_isSVGFittingCanvas", a);
  }
  async getFromFile(t) {
    const {
      lastModified: e,
      name: n,
      size: s,
      type: a
    } = t;
    return E(this, ou, dm).call(this, `${e}_${n}_${s}_${a}`, t);
  }
  async getFromUrl(t) {
    return E(this, ou, dm).call(this, t, t);
  }
  async getFromBlob(t, e) {
    const n = await e;
    return E(this, ou, dm).call(this, t, n);
  }
  async getFromId(t) {
    o(this, xn) || w(this, xn, /* @__PURE__ */ new Map());
    const e = o(this, xn).get(t);
    if (!e)
      return null;
    if (e.bitmap)
      return e.refCounter += 1, e;
    if (e.file)
      return this.getFromFile(e.file);
    if (e.blobPromise) {
      const {
        blobPromise: n
      } = e;
      return delete e.blobPromise, this.getFromBlob(e.id, n);
    }
    return this.getFromUrl(e.url);
  }
  getFromCanvas(t, e) {
    o(this, xn) || w(this, xn, /* @__PURE__ */ new Map());
    let n = o(this, xn).get(t);
    if (n != null && n.bitmap)
      return n.refCounter += 1, n;
    const s = new OffscreenCanvas(e.width, e.height);
    return s.getContext("2d").drawImage(e, 0, 0), n = {
      bitmap: s.transferToImageBitmap(),
      id: `image_${o(this, ru)}_${Ge(this, sf)._++}`,
      refCounter: 1,
      isSvg: !1
    }, o(this, xn).set(t, n), o(this, xn).set(n.id, n), n;
  }
  getSvgUrl(t) {
    const e = o(this, xn).get(t);
    return e != null && e.isSvg ? e.svgUrl : null;
  }
  deleteId(t) {
    var s;
    o(this, xn) || w(this, xn, /* @__PURE__ */ new Map());
    const e = o(this, xn).get(t);
    if (!e || (e.refCounter -= 1, e.refCounter !== 0))
      return;
    const {
      bitmap: n
    } = e;
    if (!e.url && !e.file) {
      const a = new OffscreenCanvas(n.width, n.height);
      a.getContext("bitmaprenderer").transferFromImageBitmap(n), e.blobPromise = a.convertToBlob();
    }
    (s = n.close) == null || s.call(n), e.bitmap = null;
  }
  isValidId(t) {
    return t.startsWith(`image_${o(this, ru)}_`);
  }
};
ru = new WeakMap(), sf = new WeakMap(), xn = new WeakMap(), ou = new WeakSet(), dm = async function(t, e) {
  o(this, xn) || w(this, xn, /* @__PURE__ */ new Map());
  let n = o(this, xn).get(t);
  if (n === null)
    return null;
  if (n != null && n.bitmap)
    return n.refCounter += 1, n;
  try {
    n || (n = {
      bitmap: null,
      id: `image_${o(this, ru)}_${Ge(this, sf)._++}`,
      refCounter: 0,
      isSvg: !1
    });
    let s;
    if (typeof e == "string" ? (n.url = e, s = await Lh(e, "blob")) : e instanceof File ? s = n.file = e : e instanceof Blob && (s = e), s.type === "image/svg+xml") {
      const a = _A._isSVGFittingCanvas, l = new FileReader(), u = new Image(), h = new Promise((f, g) => {
        u.onload = () => {
          n.bitmap = u, n.isSvg = !0, f();
        }, l.onload = async () => {
          const y = n.svgUrl = l.result;
          u.src = await a ? `${y}#svgView(preserveAspectRatio(none))` : y;
        }, u.onerror = l.onerror = g;
      });
      l.readAsDataURL(s), await h;
    } else
      n.bitmap = await createImageBitmap(s);
    n.refCounter = 1;
  } catch (s) {
    Rt(s), n = null;
  }
  return o(this, xn).set(t, n), n && o(this, xn).set(n.id, n), n;
};
let U0 = _A;
var Be, No, rf, Ee;
class Ek {
  constructor(t = 128) {
    b(this, Be, []);
    b(this, No, !1);
    b(this, rf);
    b(this, Ee, -1);
    w(this, rf, t);
  }
  add({
    cmd: t,
    undo: e,
    post: n,
    mustExec: s,
    type: a = NaN,
    overwriteIfSameType: l = !1,
    keepUndo: u = !1
  }) {
    if (s && t(), o(this, No))
      return;
    const h = {
      cmd: t,
      undo: e,
      post: n,
      type: a
    };
    if (o(this, Ee) === -1) {
      o(this, Be).length > 0 && (o(this, Be).length = 0), w(this, Ee, 0), o(this, Be).push(h);
      return;
    }
    if (l && o(this, Be)[o(this, Ee)].type === a) {
      u && (h.undo = o(this, Be)[o(this, Ee)].undo), o(this, Be)[o(this, Ee)] = h;
      return;
    }
    const f = o(this, Ee) + 1;
    f === o(this, rf) ? o(this, Be).splice(0, 1) : (w(this, Ee, f), f < o(this, Be).length && o(this, Be).splice(f)), o(this, Be).push(h);
  }
  undo() {
    if (o(this, Ee) === -1)
      return;
    w(this, No, !0);
    const {
      undo: t,
      post: e
    } = o(this, Be)[o(this, Ee)];
    t(), e == null || e(), w(this, No, !1), w(this, Ee, o(this, Ee) - 1);
  }
  redo() {
    if (o(this, Ee) < o(this, Be).length - 1) {
      w(this, Ee, o(this, Ee) + 1), w(this, No, !0);
      const {
        cmd: t,
        post: e
      } = o(this, Be)[o(this, Ee)];
      t(), e == null || e(), w(this, No, !1);
    }
  }
  hasSomethingToUndo() {
    return o(this, Ee) !== -1;
  }
  hasSomethingToRedo() {
    return o(this, Ee) < o(this, Be).length - 1;
  }
  cleanType(t) {
    if (o(this, Ee) !== -1) {
      for (let e = o(this, Ee); e >= 0; e--)
        if (o(this, Be)[e].type !== t) {
          o(this, Be).splice(e + 1, o(this, Ee) - e), w(this, Ee, e);
          return;
        }
      o(this, Be).length = 0, w(this, Ee, -1);
    }
  }
  destroy() {
    w(this, Be, null);
  }
}
Be = new WeakMap(), No = new WeakMap(), rf = new WeakMap(), Ee = new WeakMap();
var iy, F1;
class Bp {
  constructor(t) {
    b(this, iy);
    this.buffer = [], this.callbacks = /* @__PURE__ */ new Map(), this.allKeys = /* @__PURE__ */ new Set();
    const {
      isMac: e
    } = cn.platform;
    for (const [n, s, a = {}] of t)
      for (const l of n) {
        const u = l.startsWith("mac+");
        e && u ? (this.callbacks.set(l.slice(4), {
          callback: s,
          options: a
        }), this.allKeys.add(l.split("+").at(-1))) : !e && !u && (this.callbacks.set(l, {
          callback: s,
          options: a
        }), this.allKeys.add(l.split("+").at(-1)));
      }
  }
  exec(t, e) {
    if (!this.allKeys.has(e.key))
      return;
    const n = this.callbacks.get(E(this, iy, F1).call(this, e));
    if (!n)
      return;
    const {
      callback: s,
      options: {
        bubbles: a = !1,
        args: l = [],
        checker: u = null
      }
    } = n;
    u && !u(t, e) || (s.bind(t, ...l, e)(), a || Ce(e));
  }
}
iy = new WeakSet(), F1 = function(t) {
  t.altKey && this.buffer.push("alt"), t.ctrlKey && this.buffer.push("ctrl"), t.metaKey && this.buffer.push("meta"), t.shiftKey && this.buffer.push("shift"), this.buffer.push(t.key);
  const e = this.buffer.join("+");
  return this.buffer.length = 0, e;
};
const sy = class sy {
  get _colors() {
    const t = /* @__PURE__ */ new Map([["CanvasText", null], ["Canvas", null]]);
    return wk(t), It(this, "_colors", t);
  }
  convert(t) {
    const e = Dh(t);
    if (!window.matchMedia("(forced-colors: active)").matches)
      return e;
    for (const [n, s] of this._colors)
      if (s.every((a, l) => a === e[l]))
        return sy._colorsMapping.get(n);
    return e;
  }
  getHexCode(t) {
    const e = this._colors.get(t);
    return e ? ft.makeHexColor(...e) : t;
  }
};
Z(sy, "_colorsMapping", /* @__PURE__ */ new Map([["CanvasText", [0, 0, 0]], ["Canvas", [255, 255, 255]]]));
let H0 = sy;
var au, hi, lu, Pe, qe, cu, Wi, uu, Gi, Tn, Oo, Bo, hu, zo, zs, Xi, tl, of, af, du, lf, Us, Uo, fu, Ho, Hs, ry, jo, pu, cf, $o, el, gu, Vo, uf, je, Qt, xr, Wo, Go, hf, mu, df, Xo, js, Tr, ff, pf, qi, at, fm, j0, N1, O1, Cd, B1, z1, U1, $0, H1, V0, W0, j1, Nn, Sr, $1, V1, G0, W1, xd, X0;
const Kc = class Kc {
  constructor(t, e, n, s, a, l, u, h, f, g, y, v, S, x, _, k) {
    b(this, at);
    b(this, au, new AbortController());
    b(this, hi, null);
    b(this, lu, null);
    b(this, Pe, /* @__PURE__ */ new Map());
    b(this, qe, /* @__PURE__ */ new Map());
    b(this, cu, null);
    b(this, Wi, null);
    b(this, uu, null);
    b(this, Gi, new Ek());
    b(this, Tn, null);
    b(this, Oo, null);
    b(this, Bo, null);
    b(this, hu, 0);
    b(this, zo, /* @__PURE__ */ new Set());
    b(this, zs, null);
    b(this, Xi, null);
    b(this, tl, /* @__PURE__ */ new Set());
    Z(this, "_editorUndoBar", null);
    b(this, of, !1);
    b(this, af, !1);
    b(this, du, !1);
    b(this, lf, null);
    b(this, Us, null);
    b(this, Uo, null);
    b(this, fu, null);
    b(this, Ho, !1);
    b(this, Hs, null);
    b(this, ry, new bk());
    b(this, jo, !1);
    b(this, pu, !1);
    b(this, cf, !1);
    b(this, $o, null);
    b(this, el, null);
    b(this, gu, null);
    b(this, Vo, null);
    b(this, uf, null);
    b(this, je, Pt.NONE);
    b(this, Qt, /* @__PURE__ */ new Set());
    b(this, xr, null);
    b(this, Wo, null);
    b(this, Go, null);
    b(this, hf, null);
    b(this, mu, null);
    b(this, df, {
      isEditing: !1,
      isEmpty: !0,
      hasSomethingToUndo: !1,
      hasSomethingToRedo: !1,
      hasSelectedEditor: !1,
      hasSelectedText: !1
    });
    b(this, Xo, [0, 0]);
    b(this, js, null);
    b(this, Tr, null);
    b(this, ff, null);
    b(this, pf, null);
    b(this, qi, null);
    const T = this._signal = o(this, au).signal;
    w(this, Tr, t), w(this, ff, e), w(this, pf, n), w(this, cu, s), w(this, Tn, a), w(this, Wo, l), w(this, mu, h), this._eventBus = u, u._on("editingaction", this.onEditingAction.bind(this), {
      signal: T
    }), u._on("pagechanging", this.onPageChanging.bind(this), {
      signal: T
    }), u._on("scalechanging", this.onScaleChanging.bind(this), {
      signal: T
    }), u._on("rotationchanging", this.onRotationChanging.bind(this), {
      signal: T
    }), u._on("setpreference", this.onSetPreference.bind(this), {
      signal: T
    }), u._on("switchannotationeditorparams", (R) => this.updateParams(R.type, R.value), {
      signal: T
    }), window.addEventListener("pointerdown", () => {
      w(this, pu, !0);
    }, {
      capture: !0,
      signal: T
    }), window.addEventListener("pointerup", () => {
      w(this, pu, !1);
    }, {
      capture: !0,
      signal: T
    }), E(this, at, B1).call(this), E(this, at, j1).call(this), E(this, at, $0).call(this), w(this, Wi, h.annotationStorage), w(this, lf, h.filterFactory), w(this, Go, f), w(this, fu, g || null), w(this, of, y), w(this, af, v), w(this, du, S), w(this, uf, x || null), this.viewParameters = {
      realScale: lo.PDF_TO_CSS_UNITS,
      rotation: 0
    }, this.isShiftKeyDown = !1, this._editorUndoBar = _ || null, this._supportsPinchToZoom = k !== !1, a == null || a.setSidebarUiManager(this);
  }
  static get _keyboardManager() {
    const t = Kc.prototype, e = (l) => o(l, Tr).contains(document.activeElement) && document.activeElement.tagName !== "BUTTON" && l.hasSomethingToControl(), n = (l, {
      target: u
    }) => {
      if (u instanceof HTMLInputElement) {
        const {
          type: h
        } = u;
        return h !== "text" && h !== "number";
      }
      return !0;
    }, s = this.TRANSLATE_SMALL, a = this.TRANSLATE_BIG;
    return It(this, "_keyboardManager", new Bp([[["ctrl+a", "mac+meta+a"], t.selectAll, {
      checker: n
    }], [["ctrl+z", "mac+meta+z"], t.undo, {
      checker: n
    }], [["ctrl+y", "ctrl+shift+z", "mac+meta+shift+z", "ctrl+shift+Z", "mac+meta+shift+Z"], t.redo, {
      checker: n
    }], [["Backspace", "alt+Backspace", "ctrl+Backspace", "shift+Backspace", "mac+Backspace", "mac+alt+Backspace", "mac+ctrl+Backspace", "Delete", "ctrl+Delete", "shift+Delete", "mac+Delete"], t.delete, {
      checker: n
    }], [["Enter", "mac+Enter"], t.addNewEditorFromKeyboard, {
      checker: (l, {
        target: u
      }) => !(u instanceof HTMLButtonElement) && o(l, Tr).contains(u) && !l.isEnterHandled
    }], [[" ", "mac+ "], t.addNewEditorFromKeyboard, {
      checker: (l, {
        target: u
      }) => !(u instanceof HTMLButtonElement) && o(l, Tr).contains(document.activeElement)
    }], [["Escape", "mac+Escape"], t.unselectAll], [["ArrowLeft", "mac+ArrowLeft"], t.translateSelectedEditors, {
      args: [-s, 0],
      checker: e
    }], [["ctrl+ArrowLeft", "mac+shift+ArrowLeft"], t.translateSelectedEditors, {
      args: [-a, 0],
      checker: e
    }], [["ArrowRight", "mac+ArrowRight"], t.translateSelectedEditors, {
      args: [s, 0],
      checker: e
    }], [["ctrl+ArrowRight", "mac+shift+ArrowRight"], t.translateSelectedEditors, {
      args: [a, 0],
      checker: e
    }], [["ArrowUp", "mac+ArrowUp"], t.translateSelectedEditors, {
      args: [0, -s],
      checker: e
    }], [["ctrl+ArrowUp", "mac+shift+ArrowUp"], t.translateSelectedEditors, {
      args: [0, -a],
      checker: e
    }], [["ArrowDown", "mac+ArrowDown"], t.translateSelectedEditors, {
      args: [0, s],
      checker: e
    }], [["ctrl+ArrowDown", "mac+shift+ArrowDown"], t.translateSelectedEditors, {
      args: [0, a],
      checker: e
    }]]));
  }
  destroy() {
    var t, e, n, s, a, l, u, h, f;
    (t = o(this, qi)) == null || t.resolve(), w(this, qi, null), (e = o(this, au)) == null || e.abort(), w(this, au, null), this._signal = null;
    for (const g of o(this, qe).values())
      g.destroy();
    o(this, qe).clear(), o(this, Pe).clear(), o(this, tl).clear(), (n = o(this, Vo)) == null || n.clear(), w(this, hi, null), o(this, Qt).clear(), o(this, Gi).destroy(), (s = o(this, cu)) == null || s.destroy(), (a = o(this, Tn)) == null || a.destroy(), (l = o(this, Wo)) == null || l.destroy(), (u = o(this, Hs)) == null || u.hide(), w(this, Hs, null), (h = o(this, gu)) == null || h.destroy(), w(this, gu, null), w(this, lu, null), o(this, Us) && (clearTimeout(o(this, Us)), w(this, Us, null)), o(this, js) && (clearTimeout(o(this, js)), w(this, js, null)), (f = this._editorUndoBar) == null || f.destroy(), w(this, mu, null);
  }
  combinedSignal(t) {
    return AbortSignal.any([this._signal, t.signal]);
  }
  get mlManager() {
    return o(this, uf);
  }
  get useNewAltTextFlow() {
    return o(this, af);
  }
  get useNewAltTextWhenAddingImage() {
    return o(this, du);
  }
  get hcmFilter() {
    return It(this, "hcmFilter", o(this, Go) ? o(this, lf).addHCMFilter(o(this, Go).foreground, o(this, Go).background) : "none");
  }
  get direction() {
    return It(this, "direction", getComputedStyle(o(this, Tr)).direction);
  }
  get _highlightColors() {
    return It(this, "_highlightColors", o(this, fu) ? new Map(o(this, fu).split(",").map((t) => (t = t.split("=").map((e) => e.trim()), t[1] = t[1].toUpperCase(), t))) : null);
  }
  get highlightColors() {
    const {
      _highlightColors: t
    } = this;
    if (!t)
      return It(this, "highlightColors", null);
    const e = /* @__PURE__ */ new Map(), n = !!o(this, Go);
    for (const [s, a] of t) {
      const l = s.endsWith("_HCM");
      if (n && l) {
        e.set(s.replace("_HCM", ""), a);
        continue;
      }
      !n && !l && e.set(s, a);
    }
    return It(this, "highlightColors", e);
  }
  get highlightColorNames() {
    return It(this, "highlightColorNames", this.highlightColors ? new Map(Array.from(this.highlightColors, (t) => t.reverse())) : null);
  }
  getNonHCMColor(t) {
    if (!this._highlightColors)
      return t;
    const e = this.highlightColorNames.get(t);
    return this._highlightColors.get(e) || t;
  }
  getNonHCMColorName(t) {
    return this.highlightColorNames.get(t) || t;
  }
  setCurrentDrawingSession(t) {
    t ? (this.unselectAll(), this.disableUserSelect(!0)) : this.disableUserSelect(!1), w(this, Bo, t);
  }
  setMainHighlightColorPicker(t) {
    w(this, gu, t);
  }
  editAltText(t, e = !1) {
    var n;
    (n = o(this, cu)) == null || n.editAltText(this, t, e);
  }
  hasCommentManager() {
    return !!o(this, Tn);
  }
  editComment(t, e, n, s) {
    var a;
    (a = o(this, Tn)) == null || a.showDialog(this, t, e, n, s);
  }
  selectComment(t, e) {
    const n = o(this, qe).get(t), s = n == null ? void 0 : n.getEditorByUID(e);
    s == null || s.toggleComment(!0, !0);
  }
  updateComment(t) {
    var e;
    (e = o(this, Tn)) == null || e.updateComment(t.getData());
  }
  updatePopupColor(t) {
    var e;
    (e = o(this, Tn)) == null || e.updatePopupColor(t);
  }
  removeComment(t) {
    var e;
    (e = o(this, Tn)) == null || e.removeComments([t.uid]);
  }
  toggleComment(t, e, n = void 0) {
    var s;
    (s = o(this, Tn)) == null || s.toggleCommentPopup(t, e, n);
  }
  makeCommentColor(t, e) {
    var n;
    return t && ((n = o(this, Tn)) == null ? void 0 : n.makeCommentColor(t, e)) || null;
  }
  getCommentDialogElement() {
    var t;
    return ((t = o(this, Tn)) == null ? void 0 : t.dialogElement) || null;
  }
  async waitForEditorsRendered(t) {
    if (o(this, qe).has(t - 1))
      return;
    const {
      resolve: e,
      promise: n
    } = Promise.withResolvers(), s = (a) => {
      a.pageNumber === t && (this._eventBus._off("editorsrendered", s), e());
    };
    this._eventBus.on("editorsrendered", s), await n;
  }
  getSignature(t) {
    var e;
    (e = o(this, Wo)) == null || e.getSignature({
      uiManager: this,
      editor: t
    });
  }
  get signatureManager() {
    return o(this, Wo);
  }
  switchToMode(t, e) {
    this._eventBus.on("annotationeditormodechanged", e, {
      once: !0,
      signal: this._signal
    }), this._eventBus.dispatch("showannotationeditorui", {
      source: this,
      mode: t
    });
  }
  setPreference(t, e) {
    this._eventBus.dispatch("setpreference", {
      source: this,
      name: t,
      value: e
    });
  }
  onSetPreference({
    name: t,
    value: e
  }) {
    switch (t) {
      case "enableNewAltTextWhenAddingImage":
        w(this, du, e);
        break;
    }
  }
  onPageChanging({
    pageNumber: t
  }) {
    w(this, hu, t - 1);
  }
  focusMainContainer() {
    o(this, Tr).focus();
  }
  findParent(t, e) {
    for (const n of o(this, qe).values()) {
      const {
        x: s,
        y: a,
        width: l,
        height: u
      } = n.div.getBoundingClientRect();
      if (t >= s && t <= s + l && e >= a && e <= a + u)
        return n;
    }
    return null;
  }
  disableUserSelect(t = !1) {
    o(this, ff).classList.toggle("noUserSelect", t);
  }
  addShouldRescale(t) {
    o(this, tl).add(t);
  }
  removeShouldRescale(t) {
    o(this, tl).delete(t);
  }
  onScaleChanging({
    scale: t
  }) {
    var e;
    this.commitOrRemove(), this.viewParameters.realScale = t * lo.PDF_TO_CSS_UNITS;
    for (const n of o(this, tl))
      n.onScaleChanging();
    (e = o(this, Bo)) == null || e.onScaleChanging();
  }
  onRotationChanging({
    pagesRotation: t
  }) {
    this.commitOrRemove(), this.viewParameters.rotation = t;
  }
  highlightSelection(t = "", e = !1) {
    const n = document.getSelection();
    if (!n || n.isCollapsed)
      return;
    const {
      anchorNode: s,
      anchorOffset: a,
      focusNode: l,
      focusOffset: u
    } = n, h = n.toString(), g = E(this, at, fm).call(this, n).closest(".textLayer"), y = this.getSelectionBoxes(g);
    if (!y)
      return;
    n.empty();
    const v = E(this, at, j0).call(this, g), S = o(this, je) === Pt.NONE, x = () => {
      const _ = v == null ? void 0 : v.createAndAddNewEditor({
        x: 0,
        y: 0
      }, !1, {
        methodOfCreation: t,
        boxes: y,
        anchorNode: s,
        anchorOffset: a,
        focusNode: l,
        focusOffset: u,
        text: h
      });
      S && this.showAllEditors("highlight", !0, !0), e && (_ == null || _.editComment());
    };
    if (S) {
      this.switchToMode(Pt.HIGHLIGHT, x);
      return;
    }
    x();
  }
  commentSelection(t = "") {
    this.highlightSelection(t, !0);
  }
  getAndRemoveDataFromAnnotationStorage(t) {
    if (!o(this, Wi))
      return null;
    const e = `${Wd}${t}`, n = o(this, Wi).getRawValue(e);
    return n && o(this, Wi).remove(e), n;
  }
  addToAnnotationStorage(t) {
    !t.isEmpty() && o(this, Wi) && !o(this, Wi).has(t.id) && o(this, Wi).setValue(t.id, t);
  }
  a11yAlert(t, e = null) {
    const n = o(this, pf);
    n && (n.setAttribute("data-l10n-id", t), e ? n.setAttribute("data-l10n-args", JSON.stringify(e)) : n.removeAttribute("data-l10n-args"));
  }
  blur() {
    if (this.isShiftKeyDown = !1, o(this, Ho) && (w(this, Ho, !1), E(this, at, Cd).call(this, "main_toolbar")), !this.hasSelection)
      return;
    const {
      activeElement: t
    } = document;
    for (const e of o(this, Qt))
      if (e.div.contains(t)) {
        w(this, el, [e, t]), e._focusEventsAllowed = !1;
        break;
      }
  }
  focus() {
    if (!o(this, el))
      return;
    const [t, e] = o(this, el);
    w(this, el, null), e.addEventListener("focusin", () => {
      t._focusEventsAllowed = !0;
    }, {
      once: !0,
      signal: this._signal
    }), e.focus();
  }
  addEditListeners() {
    E(this, at, $0).call(this), E(this, at, V0).call(this);
  }
  removeEditListeners() {
    E(this, at, H1).call(this), E(this, at, W0).call(this);
  }
  dragOver(t) {
    for (const {
      type: e
    } of t.dataTransfer.items)
      for (const n of o(this, Xi))
        if (n.isHandlingMimeForPasting(e)) {
          t.dataTransfer.dropEffect = "copy", t.preventDefault();
          return;
        }
  }
  drop(t) {
    for (const e of t.dataTransfer.items)
      for (const n of o(this, Xi))
        if (n.isHandlingMimeForPasting(e.type)) {
          n.paste(e, this.currentLayer), t.preventDefault();
          return;
        }
  }
  copy(t) {
    var n;
    if (t.preventDefault(), (n = o(this, hi)) == null || n.commitOrRemove(), !this.hasSelection)
      return;
    const e = [];
    for (const s of o(this, Qt)) {
      const a = s.serialize(!0);
      a && e.push(a);
    }
    e.length !== 0 && t.clipboardData.setData("application/pdfjs", JSON.stringify(e));
  }
  cut(t) {
    this.copy(t), this.delete();
  }
  async paste(t) {
    t.preventDefault();
    const {
      clipboardData: e
    } = t;
    for (const a of e.items)
      for (const l of o(this, Xi))
        if (l.isHandlingMimeForPasting(a.type)) {
          l.paste(a, this.currentLayer);
          return;
        }
    let n = e.getData("application/pdfjs");
    if (!n)
      return;
    try {
      n = JSON.parse(n);
    } catch (a) {
      Rt(`paste: "${a.message}".`);
      return;
    }
    if (!Array.isArray(n))
      return;
    this.unselectAll();
    const s = this.currentLayer;
    try {
      const a = [];
      for (const h of n) {
        const f = await s.deserialize(h);
        if (!f)
          return;
        a.push(f);
      }
      const l = () => {
        for (const h of a)
          E(this, at, G0).call(this, h);
        E(this, at, X0).call(this, a);
      }, u = () => {
        for (const h of a)
          h.remove();
      };
      this.addCommands({
        cmd: l,
        undo: u,
        mustExec: !0
      });
    } catch (a) {
      Rt(`paste: "${a.message}".`);
    }
  }
  keydown(t) {
    !this.isShiftKeyDown && t.key === "Shift" && (this.isShiftKeyDown = !0), o(this, je) !== Pt.NONE && !this.isEditorHandlingKeyboard && Kc._keyboardManager.exec(this, t);
  }
  keyup(t) {
    this.isShiftKeyDown && t.key === "Shift" && (this.isShiftKeyDown = !1, o(this, Ho) && (w(this, Ho, !1), E(this, at, Cd).call(this, "main_toolbar")));
  }
  onEditingAction({
    name: t
  }) {
    switch (t) {
      case "undo":
      case "redo":
      case "delete":
      case "selectAll":
        this[t]();
        break;
      case "highlightSelection":
        this.highlightSelection("context_menu");
        break;
      case "commentSelection":
        this.commentSelection("context_menu");
        break;
    }
  }
  setEditingState(t) {
    t ? (E(this, at, z1).call(this), E(this, at, V0).call(this), E(this, at, Nn).call(this, {
      isEditing: o(this, je) !== Pt.NONE,
      isEmpty: E(this, at, xd).call(this),
      hasSomethingToUndo: o(this, Gi).hasSomethingToUndo(),
      hasSomethingToRedo: o(this, Gi).hasSomethingToRedo(),
      hasSelectedEditor: !1
    })) : (E(this, at, U1).call(this), E(this, at, W0).call(this), E(this, at, Nn).call(this, {
      isEditing: !1
    }), this.disableUserSelect(!1));
  }
  registerEditorTypes(t) {
    if (!o(this, Xi)) {
      w(this, Xi, t);
      for (const e of o(this, Xi))
        E(this, at, Sr).call(this, e.defaultPropertiesToUpdate);
    }
  }
  getId() {
    return o(this, ry).id;
  }
  get currentLayer() {
    return o(this, qe).get(o(this, hu));
  }
  getLayer(t) {
    return o(this, qe).get(t);
  }
  get currentPageIndex() {
    return o(this, hu);
  }
  addLayer(t) {
    o(this, qe).set(t.pageIndex, t), o(this, jo) ? t.enable() : t.disable();
  }
  removeLayer(t) {
    o(this, qe).delete(t.pageIndex);
  }
  async updateMode(t, e = null, n = !1, s = !1, a = !1) {
    var l, u, h, f, g, y;
    if (o(this, je) !== t && !(o(this, qi) && (await o(this, qi).promise, !o(this, qi)))) {
      if (w(this, qi, Promise.withResolvers()), (l = o(this, Bo)) == null || l.commitOrRemove(), o(this, je) === Pt.POPUP && ((u = o(this, Tn)) == null || u.hideSidebar()), (h = o(this, Tn)) == null || h.destroyPopup(), w(this, je, t), t === Pt.NONE) {
        this.setEditingState(!1), E(this, at, V1).call(this);
        for (const v of o(this, Pe).values())
          v.hideStandaloneCommentButton();
        (f = this._editorUndoBar) == null || f.hide(), this.toggleComment(null), o(this, qi).resolve();
        return;
      }
      for (const v of o(this, Pe).values())
        v.addStandaloneCommentButton();
      t === Pt.SIGNATURE && await ((g = o(this, Wo)) == null ? void 0 : g.loadSignatures()), this.setEditingState(!0), await E(this, at, $1).call(this), this.unselectAll();
      for (const v of o(this, qe).values())
        v.updateMode(t);
      if (t === Pt.POPUP) {
        o(this, lu) || w(this, lu, await o(this, mu).getAnnotationsByType(new Set(o(this, Xi).map((x) => x._editorType))));
        const v = /* @__PURE__ */ new Set(), S = [];
        for (const x of o(this, Pe).values()) {
          const {
            annotationElementId: _,
            hasComment: k,
            deleted: T
          } = x;
          _ && v.add(_), k && !T && S.push(x.getData());
        }
        for (const x of o(this, lu)) {
          const {
            id: _,
            popupRef: k,
            contentsObj: T
          } = x;
          k && (T != null && T.str) && !v.has(_) && !o(this, zo).has(_) && S.push(x);
        }
        (y = o(this, Tn)) == null || y.showSidebar(S);
      }
      if (!e) {
        n && this.addNewEditorFromKeyboard(), o(this, qi).resolve();
        return;
      }
      for (const v of o(this, Pe).values())
        v.uid === e ? (this.setSelected(v), a ? v.editComment() : s ? v.enterInEditMode() : v.focus()) : v.unselect();
      o(this, qi).resolve();
    }
  }
  addNewEditorFromKeyboard() {
    this.currentLayer.canCreateNewEmptyEditor() && this.currentLayer.addNewEditor();
  }
  updateToolbar(t) {
    t.mode !== o(this, je) && this._eventBus.dispatch("switchannotationeditormode", {
      source: this,
      ...t
    });
  }
  updateParams(t, e) {
    if (o(this, Xi)) {
      switch (t) {
        case jt.CREATE:
          this.currentLayer.addNewEditor(e);
          return;
        case jt.HIGHLIGHT_SHOW_ALL:
          this._eventBus.dispatch("reporttelemetry", {
            source: this,
            details: {
              type: "editing",
              data: {
                type: "highlight",
                action: "toggle_visibility"
              }
            }
          }), (o(this, hf) || w(this, hf, /* @__PURE__ */ new Map())).set(t, e), this.showAllEditors("highlight", e);
          break;
      }
      if (this.hasSelection)
        for (const n of o(this, Qt))
          n.updateParams(t, e);
      else
        for (const n of o(this, Xi))
          n.updateDefaultParams(t, e);
    }
  }
  showAllEditors(t, e, n = !1) {
    var a;
    for (const l of o(this, Pe).values())
      l.editorType === t && l.show(e);
    (((a = o(this, hf)) == null ? void 0 : a.get(jt.HIGHLIGHT_SHOW_ALL)) ?? !0) !== e && E(this, at, Sr).call(this, [[jt.HIGHLIGHT_SHOW_ALL, e]]);
  }
  enableWaiting(t = !1) {
    if (o(this, cf) !== t) {
      w(this, cf, t);
      for (const e of o(this, qe).values())
        t ? e.disableClick() : e.enableClick(), e.div.classList.toggle("waiting", t);
    }
  }
  *getEditors(t) {
    for (const e of o(this, Pe).values())
      e.pageIndex === t && (yield e);
  }
  getEditor(t) {
    return o(this, Pe).get(t);
  }
  addEditor(t) {
    o(this, Pe).set(t.id, t);
  }
  removeEditor(t) {
    var e, n;
    t.div.contains(document.activeElement) && (o(this, Us) && clearTimeout(o(this, Us)), w(this, Us, setTimeout(() => {
      this.focusMainContainer(), w(this, Us, null);
    }, 0))), o(this, Pe).delete(t.id), t.annotationElementId && ((e = o(this, Vo)) == null || e.delete(t.annotationElementId)), this.unselect(t), (!t.annotationElementId || !o(this, zo).has(t.annotationElementId)) && ((n = o(this, Wi)) == null || n.remove(t.id));
  }
  addDeletedAnnotationElement(t) {
    o(this, zo).add(t.annotationElementId), this.addChangedExistingAnnotation(t), t.deleted = !0;
  }
  isDeletedAnnotationElement(t) {
    return o(this, zo).has(t);
  }
  removeDeletedAnnotationElement(t) {
    o(this, zo).delete(t.annotationElementId), this.removeChangedExistingAnnotation(t), t.deleted = !1;
  }
  setActiveEditor(t) {
    o(this, hi) !== t && (w(this, hi, t), t && E(this, at, Sr).call(this, t.propertiesToUpdate));
  }
  updateUI(t) {
    o(this, at, W1) === t && E(this, at, Sr).call(this, t.propertiesToUpdate);
  }
  updateUIForDefaultProperties(t) {
    E(this, at, Sr).call(this, t.defaultPropertiesToUpdate);
  }
  toggleSelected(t) {
    if (o(this, Qt).has(t)) {
      o(this, Qt).delete(t), t.unselect(), E(this, at, Nn).call(this, {
        hasSelectedEditor: this.hasSelection
      });
      return;
    }
    o(this, Qt).add(t), t.select(), E(this, at, Sr).call(this, t.propertiesToUpdate), E(this, at, Nn).call(this, {
      hasSelectedEditor: !0
    });
  }
  setSelected(t) {
    var e;
    this.updateToolbar({
      mode: t.mode,
      editId: t.id
    }), (e = o(this, Bo)) == null || e.commitOrRemove();
    for (const n of o(this, Qt))
      n !== t && n.unselect();
    o(this, Qt).clear(), o(this, Qt).add(t), t.select(), E(this, at, Sr).call(this, t.propertiesToUpdate), E(this, at, Nn).call(this, {
      hasSelectedEditor: !0
    });
  }
  isSelected(t) {
    return o(this, Qt).has(t);
  }
  get firstSelectedEditor() {
    return o(this, Qt).values().next().value;
  }
  unselect(t) {
    t.unselect(), o(this, Qt).delete(t), E(this, at, Nn).call(this, {
      hasSelectedEditor: this.hasSelection
    });
  }
  get hasSelection() {
    return o(this, Qt).size !== 0;
  }
  get isEnterHandled() {
    return o(this, Qt).size === 1 && this.firstSelectedEditor.isEnterHandled;
  }
  undo() {
    var t;
    o(this, Gi).undo(), E(this, at, Nn).call(this, {
      hasSomethingToUndo: o(this, Gi).hasSomethingToUndo(),
      hasSomethingToRedo: !0,
      isEmpty: E(this, at, xd).call(this)
    }), (t = this._editorUndoBar) == null || t.hide();
  }
  redo() {
    o(this, Gi).redo(), E(this, at, Nn).call(this, {
      hasSomethingToUndo: !0,
      hasSomethingToRedo: o(this, Gi).hasSomethingToRedo(),
      isEmpty: E(this, at, xd).call(this)
    });
  }
  addCommands(t) {
    o(this, Gi).add(t), E(this, at, Nn).call(this, {
      hasSomethingToUndo: !0,
      hasSomethingToRedo: !1,
      isEmpty: E(this, at, xd).call(this)
    });
  }
  cleanUndoStack(t) {
    o(this, Gi).cleanType(t);
  }
  delete() {
    var a;
    this.commitOrRemove();
    const t = (a = this.currentLayer) == null ? void 0 : a.endDrawingSession(!0);
    if (!this.hasSelection && !t)
      return;
    const e = t ? [t] : [...o(this, Qt)], n = () => {
      var l;
      (l = this._editorUndoBar) == null || l.show(s, e.length === 1 ? e[0].editorType : e.length);
      for (const u of e)
        u.remove();
    }, s = () => {
      for (const l of e)
        E(this, at, G0).call(this, l);
    };
    this.addCommands({
      cmd: n,
      undo: s,
      mustExec: !0
    });
  }
  commitOrRemove() {
    var t;
    (t = o(this, hi)) == null || t.commitOrRemove();
  }
  hasSomethingToControl() {
    return o(this, hi) || this.hasSelection;
  }
  selectAll() {
    for (const t of o(this, Qt))
      t.commit();
    E(this, at, X0).call(this, o(this, Pe).values());
  }
  unselectAll() {
    var t;
    if (!(o(this, hi) && (o(this, hi).commitOrRemove(), o(this, je) !== Pt.NONE)) && !((t = o(this, Bo)) != null && t.commitOrRemove()) && this.hasSelection) {
      for (const e of o(this, Qt))
        e.unselect();
      o(this, Qt).clear(), E(this, at, Nn).call(this, {
        hasSelectedEditor: !1
      });
    }
  }
  translateSelectedEditors(t, e, n = !1) {
    if (n || this.commitOrRemove(), !this.hasSelection)
      return;
    o(this, Xo)[0] += t, o(this, Xo)[1] += e;
    const [s, a] = o(this, Xo), l = [...o(this, Qt)], u = 1e3;
    o(this, js) && clearTimeout(o(this, js)), w(this, js, setTimeout(() => {
      w(this, js, null), o(this, Xo)[0] = o(this, Xo)[1] = 0, this.addCommands({
        cmd: () => {
          for (const h of l)
            o(this, Pe).has(h.id) && (h.translateInPage(s, a), h.translationDone());
        },
        undo: () => {
          for (const h of l)
            o(this, Pe).has(h.id) && (h.translateInPage(-s, -a), h.translationDone());
        },
        mustExec: !1
      });
    }, u));
    for (const h of l)
      h.translateInPage(t, e), h.translationDone();
  }
  setUpDragSession() {
    if (this.hasSelection) {
      this.disableUserSelect(!0), w(this, zs, /* @__PURE__ */ new Map());
      for (const t of o(this, Qt))
        o(this, zs).set(t, {
          savedX: t.x,
          savedY: t.y,
          savedPageIndex: t.pageIndex,
          newX: 0,
          newY: 0,
          newPageIndex: -1
        });
    }
  }
  endDragSession() {
    if (!o(this, zs))
      return !1;
    this.disableUserSelect(!1);
    const t = o(this, zs);
    w(this, zs, null);
    let e = !1;
    for (const [{
      x: s,
      y: a,
      pageIndex: l
    }, u] of t)
      u.newX = s, u.newY = a, u.newPageIndex = l, e || (e = s !== u.savedX || a !== u.savedY || l !== u.savedPageIndex);
    if (!e)
      return !1;
    const n = (s, a, l, u) => {
      if (o(this, Pe).has(s.id)) {
        const h = o(this, qe).get(u);
        h ? s._setParentAndPosition(h, a, l) : (s.pageIndex = u, s.x = a, s.y = l);
      }
    };
    return this.addCommands({
      cmd: () => {
        for (const [s, {
          newX: a,
          newY: l,
          newPageIndex: u
        }] of t)
          n(s, a, l, u);
      },
      undo: () => {
        for (const [s, {
          savedX: a,
          savedY: l,
          savedPageIndex: u
        }] of t)
          n(s, a, l, u);
      },
      mustExec: !0
    }), !0;
  }
  dragSelectedEditors(t, e) {
    if (o(this, zs))
      for (const n of o(this, zs).keys())
        n.drag(t, e);
  }
  rebuild(t) {
    if (t.parent === null) {
      const e = this.getLayer(t.pageIndex);
      e ? (e.changeParent(t), e.addOrRebuild(t)) : (this.addEditor(t), this.addToAnnotationStorage(t), t.rebuild());
    } else
      t.parent.addOrRebuild(t);
  }
  get isEditorHandlingKeyboard() {
    var t;
    return ((t = this.getActive()) == null ? void 0 : t.shouldGetKeyboardEvents()) || o(this, Qt).size === 1 && this.firstSelectedEditor.shouldGetKeyboardEvents();
  }
  isActive(t) {
    return o(this, hi) === t;
  }
  getActive() {
    return o(this, hi);
  }
  getMode() {
    return o(this, je);
  }
  isEditingMode() {
    return o(this, je) !== Pt.NONE;
  }
  get imageManager() {
    return It(this, "imageManager", new U0());
  }
  getSelectionBoxes(t) {
    if (!t)
      return null;
    const e = document.getSelection();
    for (let f = 0, g = e.rangeCount; f < g; f++)
      if (!t.contains(e.getRangeAt(f).commonAncestorContainer))
        return null;
    const {
      x: n,
      y: s,
      width: a,
      height: l
    } = t.getBoundingClientRect();
    let u;
    switch (t.getAttribute("data-main-rotation")) {
      case "90":
        u = (f, g, y, v) => ({
          x: (g - s) / l,
          y: 1 - (f + y - n) / a,
          width: v / l,
          height: y / a
        });
        break;
      case "180":
        u = (f, g, y, v) => ({
          x: 1 - (f + y - n) / a,
          y: 1 - (g + v - s) / l,
          width: y / a,
          height: v / l
        });
        break;
      case "270":
        u = (f, g, y, v) => ({
          x: 1 - (g + v - s) / l,
          y: (f - n) / a,
          width: v / l,
          height: y / a
        });
        break;
      default:
        u = (f, g, y, v) => ({
          x: (f - n) / a,
          y: (g - s) / l,
          width: y / a,
          height: v / l
        });
        break;
    }
    const h = [];
    for (let f = 0, g = e.rangeCount; f < g; f++) {
      const y = e.getRangeAt(f);
      if (!y.collapsed)
        for (const {
          x: v,
          y: S,
          width: x,
          height: _
        } of y.getClientRects())
          x === 0 || _ === 0 || h.push(u(v, S, x, _));
    }
    return h.length === 0 ? null : h;
  }
  addChangedExistingAnnotation({
    annotationElementId: t,
    id: e
  }) {
    (o(this, uu) || w(this, uu, /* @__PURE__ */ new Map())).set(t, e);
  }
  removeChangedExistingAnnotation({
    annotationElementId: t
  }) {
    var e;
    (e = o(this, uu)) == null || e.delete(t);
  }
  renderAnnotationElement(t) {
    var s;
    const e = (s = o(this, uu)) == null ? void 0 : s.get(t.data.id);
    if (!e)
      return;
    const n = o(this, Wi).getRawValue(e);
    n && (o(this, je) === Pt.NONE && !n.hasBeenModified || n.renderAnnotationElement(t));
  }
  setMissingCanvas(t, e, n) {
    var a;
    const s = (a = o(this, Vo)) == null ? void 0 : a.get(t);
    s && (s.setCanvas(e, n), o(this, Vo).delete(t));
  }
  addMissingCanvas(t, e) {
    (o(this, Vo) || w(this, Vo, /* @__PURE__ */ new Map())).set(t, e);
  }
};
au = new WeakMap(), hi = new WeakMap(), lu = new WeakMap(), Pe = new WeakMap(), qe = new WeakMap(), cu = new WeakMap(), Wi = new WeakMap(), uu = new WeakMap(), Gi = new WeakMap(), Tn = new WeakMap(), Oo = new WeakMap(), Bo = new WeakMap(), hu = new WeakMap(), zo = new WeakMap(), zs = new WeakMap(), Xi = new WeakMap(), tl = new WeakMap(), of = new WeakMap(), af = new WeakMap(), du = new WeakMap(), lf = new WeakMap(), Us = new WeakMap(), Uo = new WeakMap(), fu = new WeakMap(), Ho = new WeakMap(), Hs = new WeakMap(), ry = new WeakMap(), jo = new WeakMap(), pu = new WeakMap(), cf = new WeakMap(), $o = new WeakMap(), el = new WeakMap(), gu = new WeakMap(), Vo = new WeakMap(), uf = new WeakMap(), je = new WeakMap(), Qt = new WeakMap(), xr = new WeakMap(), Wo = new WeakMap(), Go = new WeakMap(), hf = new WeakMap(), mu = new WeakMap(), df = new WeakMap(), Xo = new WeakMap(), js = new WeakMap(), Tr = new WeakMap(), ff = new WeakMap(), pf = new WeakMap(), qi = new WeakMap(), at = new WeakSet(), fm = function({
  anchorNode: t
}) {
  return t.nodeType === Node.TEXT_NODE ? t.parentElement : t;
}, j0 = function(t) {
  const {
    currentLayer: e
  } = this;
  if (e.hasTextLayer(t))
    return e;
  for (const n of o(this, qe).values())
    if (n.hasTextLayer(t))
      return n;
  return null;
}, N1 = function() {
  const t = document.getSelection();
  if (!t || t.isCollapsed)
    return;
  const n = E(this, at, fm).call(this, t).closest(".textLayer"), s = this.getSelectionBoxes(n);
  s && (o(this, Hs) || w(this, Hs, new Sk(this)), o(this, Hs).show(n, s, this.direction === "ltr"));
}, O1 = function() {
  var a, l, u;
  const t = document.getSelection();
  if (!t || t.isCollapsed) {
    o(this, xr) && ((a = o(this, Hs)) == null || a.hide(), w(this, xr, null), E(this, at, Nn).call(this, {
      hasSelectedText: !1
    }));
    return;
  }
  const {
    anchorNode: e
  } = t;
  if (e === o(this, xr))
    return;
  const s = E(this, at, fm).call(this, t).closest(".textLayer");
  if (!s) {
    o(this, xr) && ((l = o(this, Hs)) == null || l.hide(), w(this, xr, null), E(this, at, Nn).call(this, {
      hasSelectedText: !1
    }));
    return;
  }
  if ((u = o(this, Hs)) == null || u.hide(), w(this, xr, e), E(this, at, Nn).call(this, {
    hasSelectedText: !0
  }), !(o(this, je) !== Pt.HIGHLIGHT && o(this, je) !== Pt.NONE) && (o(this, je) === Pt.HIGHLIGHT && this.showAllEditors("highlight", !0, !0), w(this, Ho, this.isShiftKeyDown), !this.isShiftKeyDown)) {
    const h = o(this, je) === Pt.HIGHLIGHT ? E(this, at, j0).call(this, s) : null;
    if (h == null || h.toggleDrawing(), o(this, pu)) {
      const f = new AbortController(), g = this.combinedSignal(f), y = (v) => {
        v.type === "pointerup" && v.button !== 0 || (f.abort(), h == null || h.toggleDrawing(!0), v.type === "pointerup" && E(this, at, Cd).call(this, "main_toolbar"));
      };
      window.addEventListener("pointerup", y, {
        signal: g
      }), window.addEventListener("blur", y, {
        signal: g
      });
    } else
      h == null || h.toggleDrawing(!0), E(this, at, Cd).call(this, "main_toolbar");
  }
}, Cd = function(t = "") {
  o(this, je) === Pt.HIGHLIGHT ? this.highlightSelection(t) : o(this, of) && E(this, at, N1).call(this);
}, B1 = function() {
  document.addEventListener("selectionchange", E(this, at, O1).bind(this), {
    signal: this._signal
  });
}, z1 = function() {
  if (o(this, Uo))
    return;
  w(this, Uo, new AbortController());
  const t = this.combinedSignal(o(this, Uo));
  window.addEventListener("focus", this.focus.bind(this), {
    signal: t
  }), window.addEventListener("blur", this.blur.bind(this), {
    signal: t
  });
}, U1 = function() {
  var t;
  (t = o(this, Uo)) == null || t.abort(), w(this, Uo, null);
}, $0 = function() {
  if (o(this, $o))
    return;
  w(this, $o, new AbortController());
  const t = this.combinedSignal(o(this, $o));
  window.addEventListener("keydown", this.keydown.bind(this), {
    signal: t
  }), window.addEventListener("keyup", this.keyup.bind(this), {
    signal: t
  });
}, H1 = function() {
  var t;
  (t = o(this, $o)) == null || t.abort(), w(this, $o, null);
}, V0 = function() {
  if (o(this, Oo))
    return;
  w(this, Oo, new AbortController());
  const t = this.combinedSignal(o(this, Oo));
  document.addEventListener("copy", this.copy.bind(this), {
    signal: t
  }), document.addEventListener("cut", this.cut.bind(this), {
    signal: t
  }), document.addEventListener("paste", this.paste.bind(this), {
    signal: t
  });
}, W0 = function() {
  var t;
  (t = o(this, Oo)) == null || t.abort(), w(this, Oo, null);
}, j1 = function() {
  const t = this._signal;
  document.addEventListener("dragover", this.dragOver.bind(this), {
    signal: t
  }), document.addEventListener("drop", this.drop.bind(this), {
    signal: t
  });
}, Nn = function(t) {
  Object.entries(t).some(([n, s]) => o(this, df)[n] !== s) && (this._eventBus.dispatch("annotationeditorstateschanged", {
    source: this,
    details: Object.assign(o(this, df), t)
  }), o(this, je) === Pt.HIGHLIGHT && t.hasSelectedEditor === !1 && E(this, at, Sr).call(this, [[jt.HIGHLIGHT_FREE, !0]]));
}, Sr = function(t) {
  this._eventBus.dispatch("annotationeditorparamschanged", {
    source: this,
    details: t
  });
}, $1 = async function() {
  if (!o(this, jo)) {
    w(this, jo, !0);
    const t = [];
    for (const e of o(this, qe).values())
      t.push(e.enable());
    await Promise.all(t);
    for (const e of o(this, Pe).values())
      e.enable();
  }
}, V1 = function() {
  if (this.unselectAll(), o(this, jo)) {
    w(this, jo, !1);
    for (const t of o(this, qe).values())
      t.disable();
    for (const t of o(this, Pe).values())
      t.disable();
  }
}, G0 = function(t) {
  const e = o(this, qe).get(t.pageIndex);
  e ? e.addOrRebuild(t) : (this.addEditor(t), this.addToAnnotationStorage(t));
}, W1 = function() {
  let t = null;
  for (t of o(this, Qt))
    ;
  return t;
}, xd = function() {
  if (o(this, Pe).size === 0)
    return !0;
  if (o(this, Pe).size === 1)
    for (const t of o(this, Pe).values())
      return t.isEmpty();
  return !1;
}, X0 = function(t) {
  for (const e of o(this, Qt))
    e.unselect();
  o(this, Qt).clear();
  for (const e of t)
    e.isEmpty() || (o(this, Qt).add(e), e.select());
  E(this, at, Nn).call(this, {
    hasSelectedEditor: this.hasSelection
  });
}, Z(Kc, "TRANSLATE_SMALL", 1), Z(Kc, "TRANSLATE_BIG", 10);
let ka = Kc;
var en, $s, gs, yu, Vs, di, vu, Ws, Zn, kr, nl, Gs, qo, Ts, Td, pm;
const On = class On {
  constructor(t) {
    b(this, Ts);
    b(this, en, null);
    b(this, $s, !1);
    b(this, gs, null);
    b(this, yu, null);
    b(this, Vs, null);
    b(this, di, null);
    b(this, vu, !1);
    b(this, Ws, null);
    b(this, Zn, null);
    b(this, kr, null);
    b(this, nl, null);
    b(this, Gs, !1);
    w(this, Zn, t), w(this, Gs, t._uiManager.useNewAltTextFlow), o(On, qo) || w(On, qo, Object.freeze({
      added: "pdfjs-editor-new-alt-text-added-button",
      "added-label": "pdfjs-editor-new-alt-text-added-button-label",
      missing: "pdfjs-editor-new-alt-text-missing-button",
      "missing-label": "pdfjs-editor-new-alt-text-missing-button-label",
      review: "pdfjs-editor-new-alt-text-to-review-button",
      "review-label": "pdfjs-editor-new-alt-text-to-review-button-label"
    }));
  }
  static initialize(t) {
    On._l10n ?? (On._l10n = t);
  }
  async render() {
    const t = w(this, gs, document.createElement("button"));
    t.className = "altText", t.tabIndex = "0";
    const e = w(this, yu, document.createElement("span"));
    t.append(e), o(this, Gs) ? (t.classList.add("new"), t.setAttribute("data-l10n-id", o(On, qo).missing), e.setAttribute("data-l10n-id", o(On, qo)["missing-label"])) : (t.setAttribute("data-l10n-id", "pdfjs-editor-alt-text-button"), e.setAttribute("data-l10n-id", "pdfjs-editor-alt-text-button-label"));
    const n = o(this, Zn)._uiManager._signal;
    t.addEventListener("contextmenu", Li, {
      signal: n
    }), t.addEventListener("pointerdown", (a) => a.stopPropagation(), {
      signal: n
    });
    const s = (a) => {
      a.preventDefault(), o(this, Zn)._uiManager.editAltText(o(this, Zn)), o(this, Gs) && o(this, Zn)._reportTelemetry({
        action: "pdfjs.image.alt_text.image_status_label_clicked",
        data: {
          label: o(this, Ts, Td)
        }
      });
    };
    return t.addEventListener("click", s, {
      capture: !0,
      signal: n
    }), t.addEventListener("keydown", (a) => {
      a.target === t && a.key === "Enter" && (w(this, vu, !0), s(a));
    }, {
      signal: n
    }), await E(this, Ts, pm).call(this), t;
  }
  finish() {
    o(this, gs) && (o(this, gs).focus({
      focusVisible: o(this, vu)
    }), w(this, vu, !1));
  }
  isEmpty() {
    return o(this, Gs) ? o(this, en) === null : !o(this, en) && !o(this, $s);
  }
  hasData() {
    return o(this, Gs) ? o(this, en) !== null || !!o(this, kr) : this.isEmpty();
  }
  get guessedText() {
    return o(this, kr);
  }
  async setGuessedText(t) {
    o(this, en) === null && (w(this, kr, t), w(this, nl, await On._l10n.get("pdfjs-editor-new-alt-text-generated-alt-text-with-disclaimer", {
      generatedAltText: t
    })), E(this, Ts, pm).call(this));
  }
  toggleAltTextBadge(t = !1) {
    var e;
    if (!o(this, Gs) || o(this, en)) {
      (e = o(this, Ws)) == null || e.remove(), w(this, Ws, null);
      return;
    }
    if (!o(this, Ws)) {
      const n = w(this, Ws, document.createElement("div"));
      n.className = "noAltTextBadge", o(this, Zn).div.append(n);
    }
    o(this, Ws).classList.toggle("hidden", !t);
  }
  serialize(t) {
    let e = o(this, en);
    return !t && o(this, kr) === e && (e = o(this, nl)), {
      altText: e,
      decorative: o(this, $s),
      guessedText: o(this, kr),
      textWithDisclaimer: o(this, nl)
    };
  }
  get data() {
    return {
      altText: o(this, en),
      decorative: o(this, $s)
    };
  }
  set data({
    altText: t,
    decorative: e,
    guessedText: n,
    textWithDisclaimer: s,
    cancel: a = !1
  }) {
    n && (w(this, kr, n), w(this, nl, s)), !(o(this, en) === t && o(this, $s) === e) && (a || (w(this, en, t), w(this, $s, e)), E(this, Ts, pm).call(this));
  }
  toggle(t = !1) {
    o(this, gs) && (!t && o(this, di) && (clearTimeout(o(this, di)), w(this, di, null)), o(this, gs).disabled = !t);
  }
  shown() {
    o(this, Zn)._reportTelemetry({
      action: "pdfjs.image.alt_text.image_status_label_displayed",
      data: {
        label: o(this, Ts, Td)
      }
    });
  }
  destroy() {
    var t, e;
    (t = o(this, gs)) == null || t.remove(), w(this, gs, null), w(this, yu, null), w(this, Vs, null), (e = o(this, Ws)) == null || e.remove(), w(this, Ws, null);
  }
};
en = new WeakMap(), $s = new WeakMap(), gs = new WeakMap(), yu = new WeakMap(), Vs = new WeakMap(), di = new WeakMap(), vu = new WeakMap(), Ws = new WeakMap(), Zn = new WeakMap(), kr = new WeakMap(), nl = new WeakMap(), Gs = new WeakMap(), qo = new WeakMap(), Ts = new WeakSet(), Td = function() {
  return o(this, en) && "added" || o(this, en) === null && this.guessedText && "review" || "missing";
}, pm = async function() {
  var s, a, l;
  const t = o(this, gs);
  if (!t)
    return;
  if (o(this, Gs)) {
    if (t.classList.toggle("done", !!o(this, en)), t.setAttribute("data-l10n-id", o(On, qo)[o(this, Ts, Td)]), (s = o(this, yu)) == null || s.setAttribute("data-l10n-id", o(On, qo)[`${o(this, Ts, Td)}-label`]), !o(this, en)) {
      (a = o(this, Vs)) == null || a.remove();
      return;
    }
  } else {
    if (!o(this, en) && !o(this, $s)) {
      t.classList.remove("done"), (l = o(this, Vs)) == null || l.remove();
      return;
    }
    t.classList.add("done"), t.setAttribute("data-l10n-id", "pdfjs-editor-alt-text-edit-button");
  }
  let e = o(this, Vs);
  if (!e) {
    w(this, Vs, e = document.createElement("span")), e.className = "tooltip", e.setAttribute("role", "tooltip"), e.id = `alt-text-tooltip-${o(this, Zn).id}`;
    const u = 100, h = o(this, Zn)._uiManager._signal;
    h.addEventListener("abort", () => {
      clearTimeout(o(this, di)), w(this, di, null);
    }, {
      once: !0
    }), t.addEventListener("mouseenter", () => {
      w(this, di, setTimeout(() => {
        w(this, di, null), o(this, Vs).classList.add("show"), o(this, Zn)._reportTelemetry({
          action: "alt_text_tooltip"
        });
      }, u));
    }, {
      signal: h
    }), t.addEventListener("mouseleave", () => {
      var f;
      o(this, di) && (clearTimeout(o(this, di)), w(this, di, null)), (f = o(this, Vs)) == null || f.classList.remove("show");
    }, {
      signal: h
    });
  }
  o(this, $s) ? e.setAttribute("data-l10n-id", "pdfjs-editor-alt-text-decorative-tooltip") : (e.removeAttribute("data-l10n-id"), e.textContent = o(this, en)), e.parentNode || t.append(e);
  const n = o(this, Zn).getElementForAltText();
  n == null || n.setAttribute("aria-describedby", e.id);
}, b(On, qo, null), Z(On, "_l10n", null);
let Wm = On;
var fn, Yi, il, de, gf, sl, ms, rl, ol, al, mf, q0;
class em {
  constructor(t) {
    b(this, mf);
    b(this, fn, null);
    b(this, Yi, null);
    b(this, il, !1);
    b(this, de, null);
    b(this, gf, null);
    b(this, sl, null);
    b(this, ms, null);
    b(this, rl, null);
    b(this, ol, !1);
    b(this, al, null);
    w(this, de, t);
  }
  renderForToolbar() {
    const t = w(this, Yi, document.createElement("button"));
    return t.className = "comment", E(this, mf, q0).call(this, t, !1);
  }
  renderForStandalone() {
    const t = w(this, fn, document.createElement("button"));
    t.className = "annotationCommentButton";
    const e = o(this, de).commentButtonPosition;
    if (e) {
      const {
        style: n
      } = t;
      n.insetInlineEnd = `calc(${100 * (o(this, de)._uiManager.direction === "ltr" ? 1 - e[0] : e[0])}% - var(--comment-button-dim))`, n.top = `calc(${100 * e[1]}% - var(--comment-button-dim))`;
      const s = o(this, de).commentButtonColor;
      s && (n.backgroundColor = s);
    }
    return E(this, mf, q0).call(this, t, !0);
  }
  focusButton() {
    setTimeout(() => {
      var t;
      (t = o(this, fn) ?? o(this, Yi)) == null || t.focus();
    }, 0);
  }
  onUpdatedColor() {
    if (!o(this, fn))
      return;
    const t = o(this, de).commentButtonColor;
    t && (o(this, fn).style.backgroundColor = t), o(this, de)._uiManager.updatePopupColor(o(this, de));
  }
  get commentButtonWidth() {
    var t;
    return (((t = o(this, fn)) == null ? void 0 : t.getBoundingClientRect().width) ?? 0) / o(this, de).parent.boundingClientRect.width;
  }
  get commentPopupPositionInLayer() {
    if (o(this, al))
      return o(this, al);
    if (!o(this, fn))
      return null;
    const {
      x: t,
      y: e,
      height: n
    } = o(this, fn).getBoundingClientRect(), {
      x: s,
      y: a,
      width: l,
      height: u
    } = o(this, de).parent.boundingClientRect;
    return [(t - s) / l, (e + n - a) / u];
  }
  set commentPopupPositionInLayer(t) {
    w(this, al, t);
  }
  hasDefaultPopupPosition() {
    return o(this, al) === null;
  }
  removeStandaloneCommentButton() {
    var t;
    (t = o(this, fn)) == null || t.remove(), w(this, fn, null);
  }
  removeToolbarCommentButton() {
    var t;
    (t = o(this, Yi)) == null || t.remove(), w(this, Yi, null);
  }
  setCommentButtonStates({
    selected: t,
    hasPopup: e
  }) {
    o(this, fn) && (o(this, fn).classList.toggle("selected", t), o(this, fn).ariaExpanded = e);
  }
  edit(t) {
    const e = this.commentPopupPositionInLayer;
    let n, s;
    if (e)
      [n, s] = e;
    else {
      [n, s] = o(this, de).commentButtonPosition;
      const {
        width: g,
        height: y,
        x: v,
        y: S
      } = o(this, de);
      n = v + n * g, s = S + s * y;
    }
    const a = o(this, de).parent.boundingClientRect, {
      x: l,
      y: u,
      width: h,
      height: f
    } = a;
    o(this, de)._uiManager.editComment(o(this, de), l + n * h, u + s * f, {
      ...t,
      parentDimensions: a
    });
  }
  finish() {
    o(this, Yi) && (o(this, Yi).focus({
      focusVisible: o(this, il)
    }), w(this, il, !1));
  }
  isDeleted() {
    return o(this, ol) || o(this, ms) === "";
  }
  isEmpty() {
    return o(this, ms) === null;
  }
  hasBeenEdited() {
    return this.isDeleted() || o(this, ms) !== o(this, gf);
  }
  serialize() {
    return this.data;
  }
  get data() {
    return {
      text: o(this, ms),
      richText: o(this, sl),
      date: o(this, rl),
      deleted: this.isDeleted()
    };
  }
  set data(t) {
    if (t !== o(this, ms) && w(this, sl, null), t === null) {
      w(this, ms, ""), w(this, ol, !0);
      return;
    }
    w(this, ms, t), w(this, rl, /* @__PURE__ */ new Date()), w(this, ol, !1);
  }
  setInitialText(t, e = null) {
    w(this, gf, t), this.data = t, w(this, rl, null), w(this, sl, e);
  }
  shown() {
  }
  destroy() {
    var t, e;
    (t = o(this, Yi)) == null || t.remove(), w(this, Yi, null), (e = o(this, fn)) == null || e.remove(), w(this, fn, null), w(this, ms, ""), w(this, sl, null), w(this, rl, null), w(this, de, null), w(this, il, !1), w(this, ol, !1);
  }
}
fn = new WeakMap(), Yi = new WeakMap(), il = new WeakMap(), de = new WeakMap(), gf = new WeakMap(), sl = new WeakMap(), ms = new WeakMap(), rl = new WeakMap(), ol = new WeakMap(), al = new WeakMap(), mf = new WeakSet(), q0 = function(t, e) {
  if (!o(this, de)._uiManager.hasCommentManager())
    return null;
  t.tabIndex = "0", t.ariaHasPopup = "dialog", e ? (t.ariaControls = "commentPopup", t.setAttribute("data-l10n-id", "pdfjs-show-comment-button")) : (t.ariaControlsElements = [o(this, de)._uiManager.getCommentDialogElement()], t.setAttribute("data-l10n-id", "pdfjs-editor-edit-comment-button"));
  const n = o(this, de)._uiManager._signal;
  if (!(n instanceof AbortSignal) || n.aborted)
    return t;
  t.addEventListener("contextmenu", Li, {
    signal: n
  }), e && (t.addEventListener("focusin", (a) => {
    o(this, de)._focusEventsAllowed = !1, Ce(a);
  }, {
    capture: !0,
    signal: n
  }), t.addEventListener("focusout", (a) => {
    o(this, de)._focusEventsAllowed = !0, Ce(a);
  }, {
    capture: !0,
    signal: n
  })), t.addEventListener("pointerdown", (a) => a.stopPropagation(), {
    signal: n
  });
  const s = (a) => {
    a.preventDefault(), t === o(this, Yi) ? this.edit() : o(this, de).toggleComment(!0);
  };
  return t.addEventListener("click", s, {
    capture: !0,
    signal: n
  }), t.addEventListener("keydown", (a) => {
    a.target === t && a.key === "Enter" && (w(this, il, !0), s(a));
  }, {
    signal: n
  }), t.addEventListener("pointerenter", () => {
    o(this, de).toggleComment(!1, !0);
  }, {
    signal: n
  }), t.addEventListener("pointerleave", () => {
    o(this, de).toggleComment(!1, !1);
  }, {
    signal: n
  }), t;
};
var wu, ll, yf, vf, wf, Af, Sf, Pr, cl, Rr, ul, Mr, Pa, G1, X1, q1;
const CA = class CA {
  constructor({
    container: t,
    isPinchingDisabled: e = null,
    isPinchingStopped: n = null,
    onPinchStart: s = null,
    onPinching: a = null,
    onPinchEnd: l = null,
    signal: u
  }) {
    b(this, Pa);
    b(this, wu);
    b(this, ll, !1);
    b(this, yf, null);
    b(this, vf);
    b(this, wf);
    b(this, Af);
    b(this, Sf);
    b(this, Pr, null);
    b(this, cl);
    b(this, Rr, null);
    b(this, ul);
    b(this, Mr, null);
    w(this, wu, t), w(this, yf, n), w(this, vf, e), w(this, wf, s), w(this, Af, a), w(this, Sf, l), w(this, ul, new AbortController()), w(this, cl, AbortSignal.any([u, o(this, ul).signal])), t.addEventListener("touchstart", E(this, Pa, G1).bind(this), {
      passive: !1,
      signal: o(this, cl)
    });
  }
  get MIN_TOUCH_DISTANCE_TO_PINCH() {
    return 35 / ks.pixelRatio;
  }
  destroy() {
    var t, e;
    (t = o(this, ul)) == null || t.abort(), w(this, ul, null), (e = o(this, Pr)) == null || e.abort(), w(this, Pr, null);
  }
};
wu = new WeakMap(), ll = new WeakMap(), yf = new WeakMap(), vf = new WeakMap(), wf = new WeakMap(), Af = new WeakMap(), Sf = new WeakMap(), Pr = new WeakMap(), cl = new WeakMap(), Rr = new WeakMap(), ul = new WeakMap(), Mr = new WeakMap(), Pa = new WeakSet(), G1 = function(t) {
  var s, a, l;
  if ((s = o(this, vf)) != null && s.call(this))
    return;
  if (t.touches.length === 1) {
    if (o(this, Pr))
      return;
    const u = w(this, Pr, new AbortController()), h = AbortSignal.any([o(this, cl), u.signal]), f = o(this, wu), g = {
      capture: !0,
      signal: h,
      passive: !1
    }, y = (v) => {
      var S;
      v.pointerType === "touch" && ((S = o(this, Pr)) == null || S.abort(), w(this, Pr, null));
    };
    f.addEventListener("pointerdown", (v) => {
      v.pointerType === "touch" && (Ce(v), y(v));
    }, g), f.addEventListener("pointerup", y, g), f.addEventListener("pointercancel", y, g);
    return;
  }
  if (!o(this, Mr)) {
    w(this, Mr, new AbortController());
    const u = AbortSignal.any([o(this, cl), o(this, Mr).signal]), h = o(this, wu), f = {
      signal: u,
      capture: !1,
      passive: !1
    };
    h.addEventListener("touchmove", E(this, Pa, X1).bind(this), f);
    const g = E(this, Pa, q1).bind(this);
    h.addEventListener("touchend", g, f), h.addEventListener("touchcancel", g, f), f.capture = !0, h.addEventListener("pointerdown", Ce, f), h.addEventListener("pointermove", Ce, f), h.addEventListener("pointercancel", Ce, f), h.addEventListener("pointerup", Ce, f), (a = o(this, wf)) == null || a.call(this);
  }
  if (Ce(t), t.touches.length !== 2 || (l = o(this, yf)) != null && l.call(this)) {
    w(this, Rr, null);
    return;
  }
  let [e, n] = t.touches;
  e.identifier > n.identifier && ([e, n] = [n, e]), w(this, Rr, {
    touch0X: e.screenX,
    touch0Y: e.screenY,
    touch1X: n.screenX,
    touch1Y: n.screenY
  });
}, X1 = function(t) {
  var L;
  if (!o(this, Rr) || t.touches.length !== 2)
    return;
  Ce(t);
  let [e, n] = t.touches;
  e.identifier > n.identifier && ([e, n] = [n, e]);
  const {
    screenX: s,
    screenY: a
  } = e, {
    screenX: l,
    screenY: u
  } = n, h = o(this, Rr), {
    touch0X: f,
    touch0Y: g,
    touch1X: y,
    touch1Y: v
  } = h, S = y - f, x = v - g, _ = l - s, k = u - a, T = Math.hypot(_, k) || 1, R = Math.hypot(S, x) || 1;
  if (!o(this, ll) && Math.abs(R - T) <= CA.MIN_TOUCH_DISTANCE_TO_PINCH)
    return;
  if (h.touch0X = s, h.touch0Y = a, h.touch1X = l, h.touch1Y = u, !o(this, ll)) {
    w(this, ll, !0);
    return;
  }
  const P = [(s + l) / 2, (a + u) / 2];
  (L = o(this, Af)) == null || L.call(this, P, R, T);
}, q1 = function(t) {
  var e;
  t.touches.length >= 2 || (o(this, Mr) && (o(this, Mr).abort(), w(this, Mr, null), (e = o(this, Sf)) == null || e.call(this)), o(this, Rr) && (Ce(t), w(this, Rr, null), w(this, ll, !1)));
};
let Yd = CA;
var hl, ys, we, re, Lr, Au, Yo, bf, pn, dl, Dr, Xs, Ko, Ef, fl, fi, _f, pl, Ir, qs, Su, bu, Ki, gl, Cf, oy, Tt, Y0, xf, K0, gm, Y1, K1, Q0, mm, Z0, Q1, Z1, J1, J0, t_, tw, e_, n_, i_, ew, kd;
const Mt = class Mt {
  constructor(t) {
    b(this, Tt);
    b(this, hl, null);
    b(this, ys, null);
    b(this, we, null);
    b(this, re, null);
    b(this, Lr, null);
    b(this, Au, !1);
    b(this, Yo, null);
    b(this, bf, "");
    b(this, pn, null);
    b(this, dl, null);
    b(this, Dr, null);
    b(this, Xs, null);
    b(this, Ko, null);
    b(this, Ef, "");
    b(this, fl, !1);
    b(this, fi, null);
    b(this, _f, !1);
    b(this, pl, !1);
    b(this, Ir, !1);
    b(this, qs, null);
    b(this, Su, 0);
    b(this, bu, 0);
    b(this, Ki, null);
    b(this, gl, null);
    Z(this, "isSelected", !1);
    Z(this, "_isCopy", !1);
    Z(this, "_editToolbar", null);
    Z(this, "_initialOptions", /* @__PURE__ */ Object.create(null));
    Z(this, "_initialData", null);
    Z(this, "_isVisible", !0);
    Z(this, "_uiManager", null);
    Z(this, "_focusEventsAllowed", !0);
    b(this, Cf, !1);
    b(this, oy, Mt._zIndex++);
    this.parent = t.parent, this.id = t.id, this.width = this.height = null, this.pageIndex = t.parent.pageIndex, this.name = t.name, this.div = null, this._uiManager = t.uiManager, this.annotationElementId = null, this._willKeepAspectRatio = !1, this._initialOptions.isCentered = t.isCentered, this._structTreeParentId = null, this.annotationElementId = t.annotationElementId || null, this.creationDate = t.creationDate || /* @__PURE__ */ new Date(), this.modificationDate = t.modificationDate || null;
    const {
      rotation: e,
      rawDims: {
        pageWidth: n,
        pageHeight: s,
        pageX: a,
        pageY: l
      }
    } = this.parent.viewport;
    this.rotation = e, this.pageRotation = (360 + e - this._uiManager.viewParameters.rotation) % 360, this.pageDimensions = [n, s], this.pageTranslation = [a, l];
    const [u, h] = this.parentDimensions;
    this.x = t.x / u, this.y = t.y / h, this.isAttachedToDOM = !1, this.deleted = !1;
  }
  static get _resizerKeyboardManager() {
    const t = Mt.prototype._resizeWithKeyboard, e = ka.TRANSLATE_SMALL, n = ka.TRANSLATE_BIG;
    return It(this, "_resizerKeyboardManager", new Bp([[["ArrowLeft", "mac+ArrowLeft"], t, {
      args: [-e, 0]
    }], [["ctrl+ArrowLeft", "mac+shift+ArrowLeft"], t, {
      args: [-n, 0]
    }], [["ArrowRight", "mac+ArrowRight"], t, {
      args: [e, 0]
    }], [["ctrl+ArrowRight", "mac+shift+ArrowRight"], t, {
      args: [n, 0]
    }], [["ArrowUp", "mac+ArrowUp"], t, {
      args: [0, -e]
    }], [["ctrl+ArrowUp", "mac+shift+ArrowUp"], t, {
      args: [0, -n]
    }], [["ArrowDown", "mac+ArrowDown"], t, {
      args: [0, e]
    }], [["ctrl+ArrowDown", "mac+shift+ArrowDown"], t, {
      args: [0, n]
    }], [["Escape", "mac+Escape"], Mt.prototype._stopResizingWithKeyboard]]));
  }
  get editorType() {
    return Object.getPrototypeOf(this).constructor._type;
  }
  get mode() {
    return Object.getPrototypeOf(this).constructor._editorType;
  }
  static get isDrawer() {
    return !1;
  }
  static get _defaultLineColor() {
    return It(this, "_defaultLineColor", this._colorManager.getHexCode("CanvasText"));
  }
  static deleteAnnotationElement(t) {
    const e = new _k({
      id: t.parent.getNextId(),
      parent: t.parent,
      uiManager: t._uiManager
    });
    e.annotationElementId = t.annotationElementId, e.deleted = !0, e._uiManager.addToAnnotationStorage(e);
  }
  static initialize(t, e) {
    if (Mt._l10n ?? (Mt._l10n = t), Mt._l10nResizer || (Mt._l10nResizer = Object.freeze({
      topLeft: "pdfjs-editor-resizer-top-left",
      topMiddle: "pdfjs-editor-resizer-top-middle",
      topRight: "pdfjs-editor-resizer-top-right",
      middleRight: "pdfjs-editor-resizer-middle-right",
      bottomRight: "pdfjs-editor-resizer-bottom-right",
      bottomMiddle: "pdfjs-editor-resizer-bottom-middle",
      bottomLeft: "pdfjs-editor-resizer-bottom-left",
      middleLeft: "pdfjs-editor-resizer-middle-left"
    })), Mt._borderLineWidth !== -1)
      return;
    const n = getComputedStyle(document.documentElement);
    Mt._borderLineWidth = parseFloat(n.getPropertyValue("--outline-width")) || 0;
  }
  static updateDefaultParams(t, e) {
  }
  static get defaultPropertiesToUpdate() {
    return [];
  }
  static isHandlingMimeForPasting(t) {
    return !1;
  }
  static paste(t, e) {
    ce("Not implemented");
  }
  get propertiesToUpdate() {
    return [];
  }
  get _isDraggable() {
    return o(this, Cf);
  }
  set _isDraggable(t) {
    var e;
    w(this, Cf, t), (e = this.div) == null || e.classList.toggle("draggable", t);
  }
  get uid() {
    return this.annotationElementId || this.id;
  }
  get isEnterHandled() {
    return !0;
  }
  center() {
    const [t, e] = this.pageDimensions;
    switch (this.parentRotation) {
      case 90:
        this.x -= this.height * e / (t * 2), this.y += this.width * t / (e * 2);
        break;
      case 180:
        this.x += this.width / 2, this.y += this.height / 2;
        break;
      case 270:
        this.x += this.height * e / (t * 2), this.y -= this.width * t / (e * 2);
        break;
      default:
        this.x -= this.width / 2, this.y -= this.height / 2;
        break;
    }
    this.fixAndSetPosition();
  }
  addCommands(t) {
    this._uiManager.addCommands(t);
  }
  get currentLayer() {
    return this._uiManager.currentLayer;
  }
  setInBackground() {
    this.div.style.zIndex = 0;
  }
  setInForeground() {
    this.div.style.zIndex = o(this, oy);
  }
  setParent(t) {
    var e;
    t !== null ? (this.pageIndex = t.pageIndex, this.pageDimensions = t.pageDimensions) : (E(this, Tt, kd).call(this), (e = o(this, Xs)) == null || e.remove(), w(this, Xs, null)), this.parent = t;
  }
  focusin(t) {
    this._focusEventsAllowed && (o(this, fl) ? w(this, fl, !1) : this.parent.setSelected(this));
  }
  focusout(t) {
    var n;
    if (!this._focusEventsAllowed || !this.isAttachedToDOM)
      return;
    const e = t.relatedTarget;
    e != null && e.closest(`#${this.id}`) || (t.preventDefault(), (n = this.parent) != null && n.isMultipleSelection || this.commitOrRemove());
  }
  commitOrRemove() {
    this.isEmpty() ? this.remove() : this.commit();
  }
  commit() {
    this.isInEditMode() && this.addToAnnotationStorage();
  }
  addToAnnotationStorage() {
    this._uiManager.addToAnnotationStorage(this);
  }
  setAt(t, e, n, s) {
    const [a, l] = this.parentDimensions;
    [n, s] = this.screenToPageTranslation(n, s), this.x = (t + n) / a, this.y = (e + s) / l, this.fixAndSetPosition();
  }
  _moveAfterPaste(t, e) {
    const [n, s] = this.parentDimensions;
    this.setAt(t * n, e * s, this.width * n, this.height * s), this._onTranslated();
  }
  translate(t, e) {
    E(this, Tt, Y0).call(this, this.parentDimensions, t, e);
  }
  translateInPage(t, e) {
    o(this, fi) || w(this, fi, [this.x, this.y, this.width, this.height]), E(this, Tt, Y0).call(this, this.pageDimensions, t, e), this.div.scrollIntoView({
      block: "nearest"
    });
  }
  translationDone() {
    this._onTranslated(this.x, this.y);
  }
  drag(t, e) {
    o(this, fi) || w(this, fi, [this.x, this.y, this.width, this.height]);
    const {
      div: n,
      parentDimensions: [s, a]
    } = this;
    if (this.x += t / s, this.y += e / a, this.parent && (this.x < 0 || this.x > 1 || this.y < 0 || this.y > 1)) {
      const {
        x: y,
        y: v
      } = this.div.getBoundingClientRect();
      this.parent.findNewParent(this, y, v) && (this.x -= Math.floor(this.x), this.y -= Math.floor(this.y));
    }
    let {
      x: l,
      y: u
    } = this;
    const [h, f] = this.getBaseTranslation();
    l += h, u += f;
    const {
      style: g
    } = n;
    g.left = `${(100 * l).toFixed(2)}%`, g.top = `${(100 * u).toFixed(2)}%`, this._onTranslating(l, u), n.scrollIntoView({
      block: "nearest"
    });
  }
  _onTranslating(t, e) {
  }
  _onTranslated(t, e) {
  }
  get _hasBeenMoved() {
    return !!o(this, fi) && (o(this, fi)[0] !== this.x || o(this, fi)[1] !== this.y);
  }
  get _hasBeenResized() {
    return !!o(this, fi) && (o(this, fi)[2] !== this.width || o(this, fi)[3] !== this.height);
  }
  getBaseTranslation() {
    const [t, e] = this.parentDimensions, {
      _borderLineWidth: n
    } = Mt, s = n / t, a = n / e;
    switch (this.rotation) {
      case 90:
        return [-s, a];
      case 180:
        return [s, a];
      case 270:
        return [s, -a];
      default:
        return [-s, -a];
    }
  }
  get _mustFixPosition() {
    return !0;
  }
  fixAndSetPosition(t = this.rotation) {
    const {
      div: {
        style: e
      },
      pageDimensions: [n, s]
    } = this;
    let {
      x: a,
      y: l,
      width: u,
      height: h
    } = this;
    if (u *= n, h *= s, a *= n, l *= s, this._mustFixPosition)
      switch (t) {
        case 0:
          a = Ln(a, 0, n - u), l = Ln(l, 0, s - h);
          break;
        case 90:
          a = Ln(a, 0, n - h), l = Ln(l, u, s);
          break;
        case 180:
          a = Ln(a, u, n), l = Ln(l, h, s);
          break;
        case 270:
          a = Ln(a, h, n), l = Ln(l, 0, s - u);
          break;
      }
    this.x = a /= n, this.y = l /= s;
    const [f, g] = this.getBaseTranslation();
    a += f, l += g, e.left = `${(100 * a).toFixed(2)}%`, e.top = `${(100 * l).toFixed(2)}%`, this.moveInDOM();
  }
  screenToPageTranslation(t, e) {
    var n;
    return E(n = Mt, xf, K0).call(n, t, e, this.parentRotation);
  }
  pageTranslationToScreen(t, e) {
    var n;
    return E(n = Mt, xf, K0).call(n, t, e, 360 - this.parentRotation);
  }
  get parentScale() {
    return this._uiManager.viewParameters.realScale;
  }
  get parentRotation() {
    return (this._uiManager.viewParameters.rotation + this.pageRotation) % 360;
  }
  get parentDimensions() {
    const {
      parentScale: t,
      pageDimensions: [e, n]
    } = this;
    return [e * t, n * t];
  }
  setDims() {
    const {
      div: {
        style: t
      },
      width: e,
      height: n
    } = this;
    t.width = `${(100 * e).toFixed(2)}%`, t.height = `${(100 * n).toFixed(2)}%`;
  }
  getInitialTranslation() {
    return [0, 0];
  }
  _onResized() {
  }
  static _round(t) {
    return Math.round(t * 1e4) / 1e4;
  }
  _onResizing() {
  }
  altTextFinish() {
    var t;
    (t = o(this, we)) == null || t.finish();
  }
  get toolbarButtons() {
    return null;
  }
  async addEditToolbar() {
    if (this._editToolbar || o(this, pl))
      return this._editToolbar;
    this._editToolbar = new B0(this), this.div.append(this._editToolbar.render());
    const {
      toolbarButtons: t
    } = this;
    if (t)
      for (const [e, n] of t)
        await this._editToolbar.addButton(e, n);
    return this.hasComment || this._editToolbar.addButton("comment", this.addCommentButton()), this._editToolbar.addButton("delete"), this._editToolbar;
  }
  addCommentButtonInToolbar() {
    var t;
    (t = this._editToolbar) == null || t.addButtonBefore("comment", this.addCommentButton(), ".deleteButton");
  }
  removeCommentButtonFromToolbar() {
    var t;
    (t = this._editToolbar) == null || t.removeButton("comment");
  }
  removeEditToolbar() {
    var t, e;
    (t = this._editToolbar) == null || t.remove(), this._editToolbar = null, (e = o(this, we)) == null || e.destroy();
  }
  addContainer(t) {
    var n;
    const e = (n = this._editToolbar) == null ? void 0 : n.div;
    e ? e.before(t) : this.div.append(t);
  }
  getClientDimensions() {
    return this.div.getBoundingClientRect();
  }
  createAltText() {
    return o(this, we) || (Wm.initialize(Mt._l10n), w(this, we, new Wm(this)), o(this, hl) && (o(this, we).data = o(this, hl), w(this, hl, null))), o(this, we);
  }
  get altTextData() {
    var t;
    return (t = o(this, we)) == null ? void 0 : t.data;
  }
  set altTextData(t) {
    o(this, we) && (o(this, we).data = t);
  }
  get guessedAltText() {
    var t;
    return (t = o(this, we)) == null ? void 0 : t.guessedText;
  }
  async setGuessedAltText(t) {
    var e;
    await ((e = o(this, we)) == null ? void 0 : e.setGuessedText(t));
  }
  serializeAltText(t) {
    var e;
    return (e = o(this, we)) == null ? void 0 : e.serialize(t);
  }
  hasAltText() {
    return !!o(this, we) && !o(this, we).isEmpty();
  }
  hasAltTextData() {
    var t;
    return ((t = o(this, we)) == null ? void 0 : t.hasData()) ?? !1;
  }
  focusCommentButton() {
    var t;
    (t = o(this, re)) == null || t.focusButton();
  }
  addCommentButton() {
    return o(this, re) || w(this, re, new em(this));
  }
  addStandaloneCommentButton() {
    if (o(this, Lr)) {
      this._uiManager.isEditingMode() && o(this, Lr).classList.remove("hidden");
      return;
    }
    this.hasComment && (w(this, Lr, o(this, re).renderForStandalone()), this.div.append(o(this, Lr)));
  }
  removeStandaloneCommentButton() {
    o(this, re).removeStandaloneCommentButton(), w(this, Lr, null);
  }
  hideStandaloneCommentButton() {
    var t;
    (t = o(this, Lr)) == null || t.classList.add("hidden");
  }
  get comment() {
    const {
      data: {
        richText: t,
        text: e,
        date: n,
        deleted: s
      }
    } = o(this, re);
    return {
      text: e,
      richText: t,
      date: n,
      deleted: s,
      color: this.getNonHCMColor(),
      opacity: this.opacity ?? 1
    };
  }
  set comment(t) {
    o(this, re) || w(this, re, new em(this)), o(this, re).data = t, this.hasComment ? (this.removeCommentButtonFromToolbar(), this.addStandaloneCommentButton(), this._uiManager.updateComment(this)) : (this.addCommentButtonInToolbar(), this.removeStandaloneCommentButton(), this._uiManager.removeComment(this));
  }
  setCommentData({
    comment: t,
    popupRef: e,
    richText: n
  }) {
    if (!e || (o(this, re) || w(this, re, new em(this)), o(this, re).setInitialText(t, n), !this.annotationElementId))
      return;
    const s = this._uiManager.getAndRemoveDataFromAnnotationStorage(this.annotationElementId);
    s && this.updateFromAnnotationLayer(s);
  }
  get hasEditedComment() {
    var t;
    return (t = o(this, re)) == null ? void 0 : t.hasBeenEdited();
  }
  get hasDeletedComment() {
    var t;
    return (t = o(this, re)) == null ? void 0 : t.isDeleted();
  }
  get hasComment() {
    return !!o(this, re) && !o(this, re).isEmpty() && !o(this, re).isDeleted();
  }
  async editComment(t) {
    o(this, re) || w(this, re, new em(this)), o(this, re).edit(t);
  }
  toggleComment(t, e = void 0) {
    this.hasComment && this._uiManager.toggleComment(this, t, e);
  }
  setSelectedCommentButton(t) {
    o(this, re).setSelectedButton(t);
  }
  addComment(t) {
    if (this.hasEditedComment) {
      const [, , , s] = t.rect, [a] = this.pageDimensions, [l] = this.pageTranslation, u = l + a + 1, h = s - 100, f = u + 180;
      t.popup = {
        contents: this.comment.text,
        deleted: this.comment.deleted,
        rect: [u, h, f, s]
      };
    }
  }
  updateFromAnnotationLayer({
    popup: {
      contents: t,
      deleted: e
    }
  }) {
    o(this, re).data = e ? null : t;
  }
  get parentBoundingClientRect() {
    return this.parent.boundingClientRect;
  }
  render() {
    var l;
    const t = this.div = document.createElement("div");
    t.setAttribute("data-editor-rotation", (360 - this.rotation) % 360), t.className = this.name, t.setAttribute("id", this.id), t.tabIndex = o(this, Au) ? -1 : 0, t.setAttribute("role", "application"), this.defaultL10nId && t.setAttribute("data-l10n-id", this.defaultL10nId), this._isVisible || t.classList.add("hidden"), this.setInForeground(), E(this, Tt, tw).call(this);
    const [e, n] = this.parentDimensions;
    this.parentRotation % 180 !== 0 && (t.style.maxWidth = `${(100 * n / e).toFixed(2)}%`, t.style.maxHeight = `${(100 * e / n).toFixed(2)}%`);
    const [s, a] = this.getInitialTranslation();
    return this.translate(s, a), I1(this, t, ["keydown", "pointerdown", "dblclick"]), this.isResizable && this._uiManager._supportsPinchToZoom && (o(this, gl) || w(this, gl, new Yd({
      container: t,
      isPinchingDisabled: () => !this.isSelected,
      onPinchStart: E(this, Tt, Q1).bind(this),
      onPinching: E(this, Tt, Z1).bind(this),
      onPinchEnd: E(this, Tt, J1).bind(this),
      signal: this._uiManager._signal
    }))), this.addStandaloneCommentButton(), (l = this._uiManager._editorUndoBar) == null || l.hide(), t;
  }
  pointerdown(t) {
    const {
      isMac: e
    } = cn.platform;
    if (t.button !== 0 || t.ctrlKey && e) {
      t.preventDefault();
      return;
    }
    if (w(this, fl, !0), this._isDraggable) {
      E(this, Tt, t_).call(this, t);
      return;
    }
    E(this, Tt, J0).call(this, t);
  }
  _onStartDragging() {
  }
  _onStopDragging() {
  }
  moveInDOM() {
    o(this, qs) && clearTimeout(o(this, qs)), w(this, qs, setTimeout(() => {
      var t;
      w(this, qs, null), (t = this.parent) == null || t.moveEditorInDOM(this);
    }, 0));
  }
  _setParentAndPosition(t, e, n) {
    t.changeParent(this), this.x = e, this.y = n, this.fixAndSetPosition(), this._onTranslated();
  }
  getRect(t, e, n = this.rotation) {
    const s = this.parentScale, [a, l] = this.pageDimensions, [u, h] = this.pageTranslation, f = t / s, g = e / s, y = this.x * a, v = this.y * l, S = this.width * a, x = this.height * l;
    switch (n) {
      case 0:
        return [y + f + u, l - v - g - x + h, y + f + S + u, l - v - g + h];
      case 90:
        return [y + g + u, l - v + f + h, y + g + x + u, l - v + f + S + h];
      case 180:
        return [y - f - S + u, l - v + g + h, y - f + u, l - v + g + x + h];
      case 270:
        return [y - g - x + u, l - v - f - S + h, y - g + u, l - v - f + h];
      default:
        throw new Error("Invalid rotation");
    }
  }
  getRectInCurrentCoords(t, e) {
    const [n, s, a, l] = t, u = a - n, h = l - s;
    switch (this.rotation) {
      case 0:
        return [n, e - l, u, h];
      case 90:
        return [n, e - s, h, u];
      case 180:
        return [a, e - s, u, h];
      case 270:
        return [a, e - l, h, u];
      default:
        throw new Error("Invalid rotation");
    }
  }
  getPDFRect() {
    return this.getRect(0, 0);
  }
  getNonHCMColor() {
    return this.color && Mt._colorManager.convert(this._uiManager.getNonHCMColor(this.color));
  }
  onUpdatedColor() {
    var t;
    (t = o(this, re)) == null || t.onUpdatedColor();
  }
  getData() {
    const {
      comment: {
        text: t,
        color: e,
        date: n,
        opacity: s,
        deleted: a,
        richText: l
      },
      uid: u,
      pageIndex: h,
      creationDate: f,
      modificationDate: g
    } = this;
    return {
      id: u,
      pageIndex: h,
      rect: this.getPDFRect(),
      richText: l,
      contentsObj: {
        str: t
      },
      creationDate: f,
      modificationDate: n || g,
      popupRef: !a,
      color: e,
      opacity: s
    };
  }
  onceAdded(t) {
  }
  isEmpty() {
    return !1;
  }
  enableEditMode() {
    return this.isInEditMode() ? !1 : (this.parent.setEditingState(!1), w(this, pl, !0), !0);
  }
  disableEditMode() {
    return this.isInEditMode() ? (this.parent.setEditingState(!0), w(this, pl, !1), !0) : !1;
  }
  isInEditMode() {
    return o(this, pl);
  }
  shouldGetKeyboardEvents() {
    return o(this, Ir);
  }
  needsToBeRebuilt() {
    return this.div && !this.isAttachedToDOM;
  }
  get isOnScreen() {
    const {
      top: t,
      left: e,
      bottom: n,
      right: s
    } = this.getClientDimensions(), {
      innerHeight: a,
      innerWidth: l
    } = window;
    return e < l && s > 0 && t < a && n > 0;
  }
  rebuild() {
    E(this, Tt, tw).call(this);
  }
  rotate(t) {
  }
  resize() {
  }
  serializeDeleted() {
    var t;
    return {
      id: this.annotationElementId,
      deleted: !0,
      pageIndex: this.pageIndex,
      popupRef: ((t = this._initialData) == null ? void 0 : t.popupRef) || ""
    };
  }
  serialize(t = !1, e = null) {
    var n;
    return {
      annotationType: this.mode,
      pageIndex: this.pageIndex,
      rect: this.getPDFRect(),
      rotation: this.rotation,
      structTreeParentId: this._structTreeParentId,
      popupRef: ((n = this._initialData) == null ? void 0 : n.popupRef) || ""
    };
  }
  static async deserialize(t, e, n) {
    const s = new this.prototype.constructor({
      parent: e,
      id: e.getNextId(),
      uiManager: n,
      annotationElementId: t.annotationElementId,
      creationDate: t.creationDate,
      modificationDate: t.modificationDate
    });
    s.rotation = t.rotation, w(s, hl, t.accessibilityData), s._isCopy = t.isCopy || !1;
    const [a, l] = s.pageDimensions, [u, h, f, g] = s.getRectInCurrentCoords(t.rect, l);
    return s.x = u / a, s.y = h / l, s.width = f / a, s.height = g / l, s;
  }
  get hasBeenModified() {
    return !!this.annotationElementId && (this.deleted || this.serialize() !== null);
  }
  remove() {
    var t, e;
    if ((t = o(this, Ko)) == null || t.abort(), w(this, Ko, null), this.isEmpty() || this.commit(), this.parent ? this.parent.remove(this) : this._uiManager.removeEditor(this), o(this, qs) && (clearTimeout(o(this, qs)), w(this, qs, null)), E(this, Tt, kd).call(this), this.removeEditToolbar(), o(this, Ki)) {
      for (const n of o(this, Ki).values())
        clearTimeout(n);
      w(this, Ki, null);
    }
    this.parent = null, (e = o(this, gl)) == null || e.destroy(), w(this, gl, null);
  }
  get isResizable() {
    return !1;
  }
  makeResizable() {
    this.isResizable && (E(this, Tt, Y1).call(this), o(this, pn).classList.remove("hidden"));
  }
  get toolbarPosition() {
    return null;
  }
  get commentButtonPosition() {
    return this._uiManager.direction === "ltr" ? [1, 0] : [0, 0];
  }
  get commentButtonPositionInPage() {
    const {
      commentButtonPosition: [t, e]
    } = this, [n, s, a, l] = this.getPDFRect();
    return [Mt._round(n + (a - n) * t), Mt._round(s + (l - s) * (1 - e))];
  }
  get commentButtonColor() {
    return this._uiManager.makeCommentColor(this.getNonHCMColor(), this.opacity);
  }
  get commentPopupPosition() {
    return o(this, re).commentPopupPositionInLayer;
  }
  set commentPopupPosition(t) {
    o(this, re).commentPopupPositionInLayer = t;
  }
  hasDefaultPopupPosition() {
    return o(this, re).hasDefaultPopupPosition();
  }
  get commentButtonWidth() {
    return o(this, re).commentButtonWidth;
  }
  get elementBeforePopup() {
    return this.div;
  }
  setCommentButtonStates(t) {
    o(this, re).setCommentButtonStates(t);
  }
  keydown(t) {
    if (!this.isResizable || t.target !== this.div || t.key !== "Enter")
      return;
    this._uiManager.setSelected(this), w(this, Dr, {
      savedX: this.x,
      savedY: this.y,
      savedWidth: this.width,
      savedHeight: this.height
    });
    const e = o(this, pn).children;
    if (!o(this, ys)) {
      w(this, ys, Array.from(e));
      const l = E(this, Tt, e_).bind(this), u = E(this, Tt, n_).bind(this), h = this._uiManager._signal;
      for (const f of o(this, ys)) {
        const g = f.getAttribute("data-resizer-name");
        f.setAttribute("role", "spinbutton"), f.addEventListener("keydown", l, {
          signal: h
        }), f.addEventListener("blur", u, {
          signal: h
        }), f.addEventListener("focus", E(this, Tt, i_).bind(this, g), {
          signal: h
        }), f.setAttribute("data-l10n-id", Mt._l10nResizer[g]);
      }
    }
    const n = o(this, ys)[0];
    let s = 0;
    for (const l of e) {
      if (l === n)
        break;
      s++;
    }
    const a = (360 - this.rotation + this.parentRotation) % 360 / 90 * (o(this, ys).length / 4);
    if (a !== s) {
      if (a < s)
        for (let u = 0; u < s - a; u++)
          o(this, pn).append(o(this, pn).firstChild);
      else if (a > s)
        for (let u = 0; u < a - s; u++)
          o(this, pn).firstChild.before(o(this, pn).lastChild);
      let l = 0;
      for (const u of e) {
        const f = o(this, ys)[l++].getAttribute("data-resizer-name");
        u.setAttribute("data-l10n-id", Mt._l10nResizer[f]);
      }
    }
    E(this, Tt, ew).call(this, 0), w(this, Ir, !0), o(this, pn).firstChild.focus({
      focusVisible: !0
    }), t.preventDefault(), t.stopImmediatePropagation();
  }
  _resizeWithKeyboard(t, e) {
    o(this, Ir) && E(this, Tt, Z0).call(this, o(this, Ef), {
      deltaX: t,
      deltaY: e,
      fromKeyboard: !0
    });
  }
  _stopResizingWithKeyboard() {
    E(this, Tt, kd).call(this), this.div.focus();
  }
  select() {
    var t, e, n;
    if (this.isSelected && this._editToolbar) {
      this._editToolbar.show();
      return;
    }
    if (this.isSelected = !0, this.makeResizable(), (t = this.div) == null || t.classList.add("selectedEditor"), !this._editToolbar) {
      this.addEditToolbar().then(() => {
        var s, a;
        (s = this.div) != null && s.classList.contains("selectedEditor") && ((a = this._editToolbar) == null || a.show());
      });
      return;
    }
    (e = this._editToolbar) == null || e.show(), (n = o(this, we)) == null || n.toggleAltTextBadge(!1);
  }
  focus() {
    this.div && !this.div.contains(document.activeElement) && setTimeout(() => {
      var t;
      return (t = this.div) == null ? void 0 : t.focus({
        preventScroll: !0
      });
    }, 0);
  }
  unselect() {
    var t, e, n, s, a;
    this.isSelected && (this.isSelected = !1, (t = o(this, pn)) == null || t.classList.add("hidden"), (e = this.div) == null || e.classList.remove("selectedEditor"), (n = this.div) != null && n.contains(document.activeElement) && this._uiManager.currentLayer.div.focus({
      preventScroll: !0
    }), (s = this._editToolbar) == null || s.hide(), (a = o(this, we)) == null || a.toggleAltTextBadge(!0), this.hasComment && this._uiManager.toggleComment(this, !1, !1));
  }
  updateParams(t, e) {
  }
  disableEditing() {
  }
  enableEditing() {
  }
  get canChangeContent() {
    return !1;
  }
  enterInEditMode() {
    this.canChangeContent && (this.enableEditMode(), this.div.focus());
  }
  dblclick(t) {
    t.target.nodeName !== "BUTTON" && (this.enterInEditMode(), this.parent.updateToolbar({
      mode: this.constructor._editorType,
      editId: this.id
    }));
  }
  getElementForAltText() {
    return this.div;
  }
  get contentDiv() {
    return this.div;
  }
  get isEditing() {
    return o(this, _f);
  }
  set isEditing(t) {
    w(this, _f, t), this.parent && (t ? (this.parent.setSelected(this), this.parent.setActiveEditor(this)) : this.parent.setActiveEditor(null));
  }
  static get MIN_SIZE() {
    return 16;
  }
  static canCreateNewEmptyEditor() {
    return !0;
  }
  get telemetryInitialData() {
    return {
      action: "added"
    };
  }
  get telemetryFinalData() {
    return null;
  }
  _reportTelemetry(t, e = !1) {
    if (e) {
      o(this, Ki) || w(this, Ki, /* @__PURE__ */ new Map());
      const {
        action: n
      } = t;
      let s = o(this, Ki).get(n);
      s && clearTimeout(s), s = setTimeout(() => {
        this._reportTelemetry(t), o(this, Ki).delete(n), o(this, Ki).size === 0 && w(this, Ki, null);
      }, Mt._telemetryTimeout), o(this, Ki).set(n, s);
      return;
    }
    t.type || (t.type = this.editorType), this._uiManager._eventBus.dispatch("reporttelemetry", {
      source: this,
      details: {
        type: "editing",
        data: t
      }
    });
  }
  show(t = this._isVisible) {
    this.div.classList.toggle("hidden", !t), this._isVisible = t;
  }
  enable() {
    this.div && (this.div.tabIndex = 0), w(this, Au, !1);
  }
  disable() {
    this.div && (this.div.tabIndex = -1), w(this, Au, !0);
  }
  updateFakeAnnotationElement(t) {
    if (!o(this, Xs) && !this.deleted) {
      w(this, Xs, t.addFakeAnnotation(this));
      return;
    }
    if (this.deleted) {
      o(this, Xs).remove(), w(this, Xs, null);
      return;
    }
    (this.hasEditedComment || this._hasBeenMoved || this._hasBeenResized) && o(this, Xs).updateEdited({
      rect: this.getPDFRect(),
      popup: this.comment
    });
  }
  renderAnnotationElement(t) {
    if (this.deleted)
      return t.hide(), null;
    let e = t.container.querySelector(".annotationContent");
    if (!e)
      e = document.createElement("div"), e.classList.add("annotationContent", this.editorType), t.container.prepend(e);
    else if (e.nodeName === "CANVAS") {
      const n = e;
      e = document.createElement("div"), e.classList.add("annotationContent", this.editorType), n.before(e);
    }
    return e;
  }
  resetAnnotationElement(t) {
    const {
      firstChild: e
    } = t.container;
    (e == null ? void 0 : e.nodeName) === "DIV" && e.classList.contains("annotationContent") && e.remove();
  }
};
hl = new WeakMap(), ys = new WeakMap(), we = new WeakMap(), re = new WeakMap(), Lr = new WeakMap(), Au = new WeakMap(), Yo = new WeakMap(), bf = new WeakMap(), pn = new WeakMap(), dl = new WeakMap(), Dr = new WeakMap(), Xs = new WeakMap(), Ko = new WeakMap(), Ef = new WeakMap(), fl = new WeakMap(), fi = new WeakMap(), _f = new WeakMap(), pl = new WeakMap(), Ir = new WeakMap(), qs = new WeakMap(), Su = new WeakMap(), bu = new WeakMap(), Ki = new WeakMap(), gl = new WeakMap(), Cf = new WeakMap(), oy = new WeakMap(), Tt = new WeakSet(), Y0 = function([t, e], n, s) {
  [n, s] = this.screenToPageTranslation(n, s), this.x += n / t, this.y += s / e, this._onTranslating(this.x, this.y), this.fixAndSetPosition();
}, xf = new WeakSet(), K0 = function(t, e, n) {
  switch (n) {
    case 90:
      return [e, -t];
    case 180:
      return [-t, -e];
    case 270:
      return [-e, t];
    default:
      return [t, e];
  }
}, gm = function(t) {
  switch (t) {
    case 90: {
      const [e, n] = this.pageDimensions;
      return [0, -e / n, n / e, 0];
    }
    case 180:
      return [-1, 0, 0, -1];
    case 270: {
      const [e, n] = this.pageDimensions;
      return [0, e / n, -n / e, 0];
    }
    default:
      return [1, 0, 0, 1];
  }
}, Y1 = function() {
  if (o(this, pn))
    return;
  w(this, pn, document.createElement("div")), o(this, pn).classList.add("resizers");
  const t = this._willKeepAspectRatio ? ["topLeft", "topRight", "bottomRight", "bottomLeft"] : ["topLeft", "topMiddle", "topRight", "middleRight", "bottomRight", "bottomMiddle", "bottomLeft", "middleLeft"], e = this._uiManager._signal;
  for (const n of t) {
    const s = document.createElement("div");
    o(this, pn).append(s), s.classList.add("resizer", n), s.setAttribute("data-resizer-name", n), s.addEventListener("pointerdown", E(this, Tt, K1).bind(this, n), {
      signal: e
    }), s.addEventListener("contextmenu", Li, {
      signal: e
    }), s.tabIndex = -1;
  }
  this.div.prepend(o(this, pn));
}, K1 = function(t, e) {
  var g;
  e.preventDefault();
  const {
    isMac: n
  } = cn.platform;
  if (e.button !== 0 || e.ctrlKey && n)
    return;
  (g = o(this, we)) == null || g.toggle(!1);
  const s = this._isDraggable;
  this._isDraggable = !1, w(this, dl, [e.screenX, e.screenY]);
  const a = new AbortController(), l = this._uiManager.combinedSignal(a);
  this.parent.togglePointerEvents(!1), window.addEventListener("pointermove", E(this, Tt, Z0).bind(this, t), {
    passive: !0,
    capture: !0,
    signal: l
  }), window.addEventListener("touchmove", Ce, {
    passive: !1,
    signal: l
  }), window.addEventListener("contextmenu", Li, {
    signal: l
  }), w(this, Dr, {
    savedX: this.x,
    savedY: this.y,
    savedWidth: this.width,
    savedHeight: this.height
  });
  const u = this.parent.div.style.cursor, h = this.div.style.cursor;
  this.div.style.cursor = this.parent.div.style.cursor = window.getComputedStyle(e.target).cursor;
  const f = () => {
    var y;
    a.abort(), this.parent.togglePointerEvents(!0), (y = o(this, we)) == null || y.toggle(!0), this._isDraggable = s, this.parent.div.style.cursor = u, this.div.style.cursor = h, E(this, Tt, mm).call(this);
  };
  window.addEventListener("pointerup", f, {
    signal: l
  }), window.addEventListener("blur", f, {
    signal: l
  });
}, Q0 = function(t, e, n, s) {
  this.width = n, this.height = s, this.x = t, this.y = e, this.setDims(), this.fixAndSetPosition(), this._onResized();
}, mm = function() {
  if (!o(this, Dr))
    return;
  const {
    savedX: t,
    savedY: e,
    savedWidth: n,
    savedHeight: s
  } = o(this, Dr);
  w(this, Dr, null);
  const a = this.x, l = this.y, u = this.width, h = this.height;
  a === t && l === e && u === n && h === s || this.addCommands({
    cmd: E(this, Tt, Q0).bind(this, a, l, u, h),
    undo: E(this, Tt, Q0).bind(this, t, e, n, s),
    mustExec: !0
  });
}, Z0 = function(t, e) {
  const [n, s] = this.parentDimensions, a = this.x, l = this.y, u = this.width, h = this.height, f = Mt.MIN_SIZE / n, g = Mt.MIN_SIZE / s, y = E(this, Tt, gm).call(this, this.rotation), v = (tt, Q) => [y[0] * tt + y[2] * Q, y[1] * tt + y[3] * Q], S = E(this, Tt, gm).call(this, 360 - this.rotation), x = (tt, Q) => [S[0] * tt + S[2] * Q, S[1] * tt + S[3] * Q];
  let _, k, T = !1, R = !1;
  switch (t) {
    case "topLeft":
      T = !0, _ = (tt, Q) => [0, 0], k = (tt, Q) => [tt, Q];
      break;
    case "topMiddle":
      _ = (tt, Q) => [tt / 2, 0], k = (tt, Q) => [tt / 2, Q];
      break;
    case "topRight":
      T = !0, _ = (tt, Q) => [tt, 0], k = (tt, Q) => [0, Q];
      break;
    case "middleRight":
      R = !0, _ = (tt, Q) => [tt, Q / 2], k = (tt, Q) => [0, Q / 2];
      break;
    case "bottomRight":
      T = !0, _ = (tt, Q) => [tt, Q], k = (tt, Q) => [0, 0];
      break;
    case "bottomMiddle":
      _ = (tt, Q) => [tt / 2, Q], k = (tt, Q) => [tt / 2, 0];
      break;
    case "bottomLeft":
      T = !0, _ = (tt, Q) => [0, Q], k = (tt, Q) => [tt, 0];
      break;
    case "middleLeft":
      R = !0, _ = (tt, Q) => [0, Q / 2], k = (tt, Q) => [tt, Q / 2];
      break;
  }
  const P = _(u, h), L = k(u, h);
  let D = v(...L);
  const N = Mt._round(a + D[0]), B = Mt._round(l + D[1]);
  let $ = 1, V = 1, X, q;
  if (e.fromKeyboard)
    ({
      deltaX: X,
      deltaY: q
    } = e);
  else {
    const {
      screenX: tt,
      screenY: Q
    } = e, [W, ot] = o(this, dl);
    [X, q] = this.screenToPageTranslation(tt - W, Q - ot), o(this, dl)[0] = tt, o(this, dl)[1] = Q;
  }
  if ([X, q] = x(X / n, q / s), T) {
    const tt = Math.hypot(u, h);
    $ = V = Math.max(Math.min(Math.hypot(L[0] - P[0] - X, L[1] - P[1] - q) / tt, 1 / u, 1 / h), f / u, g / h);
  } else R ? $ = Ln(Math.abs(L[0] - P[0] - X), f, 1) / u : V = Ln(Math.abs(L[1] - P[1] - q), g, 1) / h;
  const it = Mt._round(u * $), lt = Mt._round(h * V);
  D = v(...k(it, lt));
  const rt = N - D[0], vt = B - D[1];
  o(this, fi) || w(this, fi, [this.x, this.y, this.width, this.height]), this.width = it, this.height = lt, this.x = rt, this.y = vt, this.setDims(), this.fixAndSetPosition(), this._onResizing();
}, Q1 = function() {
  var t;
  w(this, Dr, {
    savedX: this.x,
    savedY: this.y,
    savedWidth: this.width,
    savedHeight: this.height
  }), (t = o(this, we)) == null || t.toggle(!1), this.parent.togglePointerEvents(!1);
}, Z1 = function(t, e, n) {
  let a = 0.7 * (n / e) + 1 - 0.7;
  if (a === 1)
    return;
  const l = E(this, Tt, gm).call(this, this.rotation), u = (N, B) => [l[0] * N + l[2] * B, l[1] * N + l[3] * B], [h, f] = this.parentDimensions, g = this.x, y = this.y, v = this.width, S = this.height, x = Mt.MIN_SIZE / h, _ = Mt.MIN_SIZE / f;
  a = Math.max(Math.min(a, 1 / v, 1 / S), x / v, _ / S);
  const k = Mt._round(v * a), T = Mt._round(S * a);
  if (k === v && T === S)
    return;
  o(this, fi) || w(this, fi, [g, y, v, S]);
  const R = u(v / 2, S / 2), P = Mt._round(g + R[0]), L = Mt._round(y + R[1]), D = u(k / 2, T / 2);
  this.x = P - D[0], this.y = L - D[1], this.width = k, this.height = T, this.setDims(), this.fixAndSetPosition(), this._onResizing();
}, J1 = function() {
  var t;
  (t = o(this, we)) == null || t.toggle(!0), this.parent.togglePointerEvents(!0), E(this, Tt, mm).call(this);
}, J0 = function(t) {
  const {
    isMac: e
  } = cn.platform;
  t.ctrlKey && !e || t.shiftKey || t.metaKey && e ? this.parent.toggleSelected(this) : this.parent.setSelected(this);
}, t_ = function(t) {
  const {
    isSelected: e
  } = this;
  this._uiManager.setUpDragSession();
  let n = !1;
  const s = new AbortController(), a = this._uiManager.combinedSignal(s), l = {
    capture: !0,
    passive: !1,
    signal: a
  }, u = (f) => {
    s.abort(), w(this, Yo, null), w(this, fl, !1), this._uiManager.endDragSession() || E(this, Tt, J0).call(this, f), n && this._onStopDragging();
  };
  e && (w(this, Su, t.clientX), w(this, bu, t.clientY), w(this, Yo, t.pointerId), w(this, bf, t.pointerType), window.addEventListener("pointermove", (f) => {
    n || (n = !0, this._uiManager.toggleComment(this, !0, !1), this._onStartDragging());
    const {
      clientX: g,
      clientY: y,
      pointerId: v
    } = f;
    if (v !== o(this, Yo)) {
      Ce(f);
      return;
    }
    const [S, x] = this.screenToPageTranslation(g - o(this, Su), y - o(this, bu));
    w(this, Su, g), w(this, bu, y), this._uiManager.dragSelectedEditors(S, x);
  }, l), window.addEventListener("touchmove", Ce, l), window.addEventListener("pointerdown", (f) => {
    f.pointerType === o(this, bf) && (o(this, gl) || f.isPrimary) && u(f), Ce(f);
  }, l));
  const h = (f) => {
    if (!o(this, Yo) || o(this, Yo) === f.pointerId) {
      u(f);
      return;
    }
    Ce(f);
  };
  window.addEventListener("pointerup", h, {
    signal: a
  }), window.addEventListener("blur", h, {
    signal: a
  });
}, tw = function() {
  if (o(this, Ko) || !this.div)
    return;
  w(this, Ko, new AbortController());
  const t = this._uiManager.combinedSignal(o(this, Ko));
  this.div.addEventListener("focusin", this.focusin.bind(this), {
    signal: t
  }), this.div.addEventListener("focusout", this.focusout.bind(this), {
    signal: t
  });
}, e_ = function(t) {
  Mt._resizerKeyboardManager.exec(this, t);
}, n_ = function(t) {
  var e;
  o(this, Ir) && ((e = t.relatedTarget) == null ? void 0 : e.parentNode) !== o(this, pn) && E(this, Tt, kd).call(this);
}, i_ = function(t) {
  w(this, Ef, o(this, Ir) ? t : "");
}, ew = function(t) {
  if (o(this, ys))
    for (const e of o(this, ys))
      e.tabIndex = t;
}, kd = function() {
  w(this, Ir, !1), E(this, Tt, ew).call(this, -1), E(this, Tt, mm).call(this);
}, b(Mt, xf), Z(Mt, "_l10n", null), Z(Mt, "_l10nResizer", null), Z(Mt, "_borderLineWidth", -1), Z(Mt, "_colorManager", new H0()), Z(Mt, "_zIndex", 1), Z(Mt, "_telemetryTimeout", 1e3);
let fe = Mt;
class _k extends fe {
  constructor(t) {
    super(t), this.annotationElementId = t.annotationElementId, this.deleted = !0;
  }
  serialize() {
    return this.serializeDeleted();
  }
}
const NE = 3285377520, Hi = 4294901760, Os = 65535;
class s_ {
  constructor(t) {
    this.h1 = t ? t & 4294967295 : NE, this.h2 = t ? t & 4294967295 : NE;
  }
  update(t) {
    let e, n;
    if (typeof t == "string") {
      e = new Uint8Array(t.length * 2), n = 0;
      for (let _ = 0, k = t.length; _ < k; _++) {
        const T = t.charCodeAt(_);
        T <= 255 ? e[n++] = T : (e[n++] = T >>> 8, e[n++] = T & 255);
      }
    } else if (ArrayBuffer.isView(t))
      e = t.slice(), n = e.byteLength;
    else
      throw new Error("Invalid data format, must be a string or TypedArray.");
    const s = n >> 2, a = n - s * 4, l = new Uint32Array(e.buffer, 0, s);
    let u = 0, h = 0, f = this.h1, g = this.h2;
    const y = 3432918353, v = 461845907, S = y & Os, x = v & Os;
    for (let _ = 0; _ < s; _++)
      _ & 1 ? (u = l[_], u = u * y & Hi | u * S & Os, u = u << 15 | u >>> 17, u = u * v & Hi | u * x & Os, f ^= u, f = f << 13 | f >>> 19, f = f * 5 + 3864292196) : (h = l[_], h = h * y & Hi | h * S & Os, h = h << 15 | h >>> 17, h = h * v & Hi | h * x & Os, g ^= h, g = g << 13 | g >>> 19, g = g * 5 + 3864292196);
    switch (u = 0, a) {
      case 3:
        u ^= e[s * 4 + 2] << 16;
      case 2:
        u ^= e[s * 4 + 1] << 8;
      case 1:
        u ^= e[s * 4], u = u * y & Hi | u * S & Os, u = u << 15 | u >>> 17, u = u * v & Hi | u * x & Os, s & 1 ? f ^= u : g ^= u;
    }
    this.h1 = f, this.h2 = g;
  }
  hexdigest() {
    let t = this.h1, e = this.h2;
    return t ^= e >>> 1, t = t * 3981806797 & Hi | t * 36045 & Os, e = e * 4283543511 & Hi | ((e << 16 | t >>> 16) * 2950163797 & Hi) >>> 16, t ^= e >>> 1, t = t * 444984403 & Hi | t * 60499 & Os, e = e * 3301882366 & Hi | ((e << 16 | t >>> 16) * 3120437893 & Hi) >>> 16, t ^= e >>> 1, (t >>> 0).toString(16).padStart(8, "0") + (e >>> 0).toString(16).padStart(8, "0");
  }
}
const nw = Object.freeze({
  map: null,
  hash: "",
  transfer: void 0
});
var ml, yl, Fr, gn, ay, r_;
class pA {
  constructor() {
    b(this, ay);
    b(this, ml, !1);
    b(this, yl, null);
    b(this, Fr, null);
    b(this, gn, /* @__PURE__ */ new Map());
    this.onSetModified = null, this.onResetModified = null, this.onAnnotationEditor = null;
  }
  getValue(t, e) {
    const n = o(this, gn).get(t);
    return n === void 0 ? e : Object.assign(e, n);
  }
  getRawValue(t) {
    return o(this, gn).get(t);
  }
  remove(t) {
    const e = o(this, gn).get(t);
    if (e !== void 0 && (e instanceof fe && o(this, Fr).delete(e.annotationElementId), o(this, gn).delete(t), o(this, gn).size === 0 && this.resetModified(), typeof this.onAnnotationEditor == "function")) {
      for (const n of o(this, gn).values())
        if (n instanceof fe)
          return;
      this.onAnnotationEditor(null);
    }
  }
  setValue(t, e) {
    const n = o(this, gn).get(t);
    let s = !1;
    if (n !== void 0)
      for (const [a, l] of Object.entries(e))
        n[a] !== l && (s = !0, n[a] = l);
    else
      s = !0, o(this, gn).set(t, e);
    s && E(this, ay, r_).call(this), e instanceof fe && ((o(this, Fr) || w(this, Fr, /* @__PURE__ */ new Map())).set(e.annotationElementId, e), typeof this.onAnnotationEditor == "function" && this.onAnnotationEditor(e.constructor._type));
  }
  has(t) {
    return o(this, gn).has(t);
  }
  get size() {
    return o(this, gn).size;
  }
  resetModified() {
    o(this, ml) && (w(this, ml, !1), typeof this.onResetModified == "function" && this.onResetModified());
  }
  get print() {
    return new o_(this);
  }
  get serializable() {
    if (o(this, gn).size === 0)
      return nw;
    const t = /* @__PURE__ */ new Map(), e = new s_(), n = [], s = /* @__PURE__ */ Object.create(null);
    let a = !1;
    for (const [l, u] of o(this, gn)) {
      const h = u instanceof fe ? u.serialize(!1, s) : u;
      h && (t.set(l, h), e.update(`${l}:${JSON.stringify(h)}`), a || (a = !!h.bitmap));
    }
    if (a)
      for (const l of t.values())
        l.bitmap && n.push(l.bitmap);
    return t.size > 0 ? {
      map: t,
      hash: e.hexdigest(),
      transfer: n
    } : nw;
  }
  get editorStats() {
    let t = null;
    const e = /* @__PURE__ */ new Map();
    let n = 0, s = 0;
    for (const a of o(this, gn).values()) {
      if (!(a instanceof fe)) {
        a.popup && (a.popup.deleted ? s += 1 : n += 1);
        continue;
      }
      a.isCommentDeleted ? s += 1 : a.hasEditedComment && (n += 1);
      const l = a.telemetryFinalData;
      if (!l)
        continue;
      const {
        type: u
      } = l;
      e.has(u) || e.set(u, Object.getPrototypeOf(a).constructor), t || (t = /* @__PURE__ */ Object.create(null));
      const h = t[u] || (t[u] = /* @__PURE__ */ new Map());
      for (const [f, g] of Object.entries(l)) {
        if (f === "type")
          continue;
        let y = h.get(f);
        y || (y = /* @__PURE__ */ new Map(), h.set(f, y));
        const v = y.get(g) ?? 0;
        y.set(g, v + 1);
      }
    }
    if ((s > 0 || n > 0) && (t || (t = /* @__PURE__ */ Object.create(null)), t.comments = {
      deleted: s,
      edited: n
    }), !t)
      return null;
    for (const [a, l] of e)
      t[a] = l.computeTelemetryFinalData(t[a]);
    return t;
  }
  resetModifiedIds() {
    w(this, yl, null);
  }
  updateEditor(t, e) {
    var s;
    const n = (s = o(this, Fr)) == null ? void 0 : s.get(t);
    return n ? (n.updateFromAnnotationLayer(e), !0) : !1;
  }
  getEditor(t) {
    var e;
    return ((e = o(this, Fr)) == null ? void 0 : e.get(t)) || null;
  }
  get modifiedIds() {
    if (o(this, yl))
      return o(this, yl);
    const t = [];
    if (o(this, Fr))
      for (const e of o(this, Fr).values())
        e.serialize() && t.push(e.annotationElementId);
    return w(this, yl, {
      ids: new Set(t),
      hash: t.join(",")
    });
  }
  [Symbol.iterator]() {
    return o(this, gn).entries();
  }
}
ml = new WeakMap(), yl = new WeakMap(), Fr = new WeakMap(), gn = new WeakMap(), ay = new WeakSet(), r_ = function() {
  o(this, ml) || (w(this, ml, !0), typeof this.onSetModified == "function" && this.onSetModified());
};
var Tf;
class o_ extends pA {
  constructor(e) {
    super();
    b(this, Tf);
    const {
      map: n,
      hash: s,
      transfer: a
    } = e.serializable, l = structuredClone(n, a ? {
      transfer: a
    } : null);
    w(this, Tf, {
      map: l,
      hash: s,
      transfer: a
    });
  }
  get print() {
    ce("Should not call PrintAnnotationStorage.print");
  }
  get serializable() {
    return o(this, Tf);
  }
  get modifiedIds() {
    return It(this, "modifiedIds", {
      ids: /* @__PURE__ */ new Set(),
      hash: ""
    });
  }
}
Tf = new WeakMap();
var Eu;
class Ck {
  constructor({
    ownerDocument: t = globalThis.document,
    styleElement: e = null
  }) {
    b(this, Eu, /* @__PURE__ */ new Set());
    this._document = t, this.nativeFontFaces = /* @__PURE__ */ new Set(), this.styleElement = null, this.loadingRequests = [], this.loadTestFontId = 0;
  }
  addNativeFontFace(t) {
    this.nativeFontFaces.add(t), this._document.fonts.add(t);
  }
  removeNativeFontFace(t) {
    this.nativeFontFaces.delete(t), this._document.fonts.delete(t);
  }
  insertRule(t) {
    this.styleElement || (this.styleElement = this._document.createElement("style"), this._document.documentElement.getElementsByTagName("head")[0].append(this.styleElement));
    const e = this.styleElement.sheet;
    e.insertRule(t, e.cssRules.length);
  }
  clear() {
    for (const t of this.nativeFontFaces)
      this._document.fonts.delete(t);
    this.nativeFontFaces.clear(), o(this, Eu).clear(), this.styleElement && (this.styleElement.remove(), this.styleElement = null);
  }
  async loadSystemFont({
    systemFontInfo: t,
    disableFontFace: e,
    _inspectFont: n
  }) {
    if (!(!t || o(this, Eu).has(t.loadedName))) {
      if ($t(!e, "loadSystemFont shouldn't be called when `disableFontFace` is set."), this.isFontLoadingAPISupported) {
        const {
          loadedName: s,
          src: a,
          style: l
        } = t, u = new FontFace(s, a, l);
        this.addNativeFontFace(u);
        try {
          await u.load(), o(this, Eu).add(s), n == null || n(t);
        } catch {
          Rt(`Cannot load system font: ${t.baseFontName}, installing it could help to improve PDF rendering.`), this.removeNativeFontFace(u);
        }
        return;
      }
      ce("Not implemented: loadSystemFont without the Font Loading API.");
    }
  }
  async bind(t) {
    if (t.attached || t.missingFile && !t.systemFontInfo)
      return;
    if (t.attached = !0, t.systemFontInfo) {
      await this.loadSystemFont(t);
      return;
    }
    if (this.isFontLoadingAPISupported) {
      const n = t.createNativeFontFace();
      if (n) {
        this.addNativeFontFace(n);
        try {
          await n.loaded;
        } catch (s) {
          throw Rt(`Failed to load font '${n.family}': '${s}'.`), t.disableFontFace = !0, s;
        }
      }
      return;
    }
    const e = t.createFontFaceRule();
    if (e) {
      if (this.insertRule(e), this.isSyncFontLoadingSupported)
        return;
      await new Promise((n) => {
        const s = this._queueLoadingCallback(n);
        this._prepareFontLoadEvent(t, s);
      });
    }
  }
  get isFontLoadingAPISupported() {
    var e;
    const t = !!((e = this._document) != null && e.fonts);
    return It(this, "isFontLoadingAPISupported", t);
  }
  get isSyncFontLoadingSupported() {
    return It(this, "isSyncFontLoadingSupported", Hn || cn.platform.isFirefox);
  }
  _queueLoadingCallback(t) {
    function e() {
      for ($t(!s.done, "completeRequest() cannot be called twice."), s.done = !0; n.length > 0 && n[0].done; ) {
        const a = n.shift();
        setTimeout(a.callback, 0);
      }
    }
    const {
      loadingRequests: n
    } = this, s = {
      done: !1,
      complete: e,
      callback: t
    };
    return n.push(s), s;
  }
  get _loadTestFont() {
    const t = atob("T1RUTwALAIAAAwAwQ0ZGIDHtZg4AAAOYAAAAgUZGVE1lkzZwAAAEHAAAABxHREVGABQAFQAABDgAAAAeT1MvMlYNYwkAAAEgAAAAYGNtYXABDQLUAAACNAAAAUJoZWFk/xVFDQAAALwAAAA2aGhlYQdkA+oAAAD0AAAAJGhtdHgD6AAAAAAEWAAAAAZtYXhwAAJQAAAAARgAAAAGbmFtZVjmdH4AAAGAAAAAsXBvc3T/hgAzAAADeAAAACAAAQAAAAEAALZRFsRfDzz1AAsD6AAAAADOBOTLAAAAAM4KHDwAAAAAA+gDIQAAAAgAAgAAAAAAAAABAAADIQAAAFoD6AAAAAAD6AABAAAAAAAAAAAAAAAAAAAAAQAAUAAAAgAAAAQD6AH0AAUAAAKKArwAAACMAooCvAAAAeAAMQECAAACAAYJAAAAAAAAAAAAAQAAAAAAAAAAAAAAAFBmRWQAwAAuAC4DIP84AFoDIQAAAAAAAQAAAAAAAAAAACAAIAABAAAADgCuAAEAAAAAAAAAAQAAAAEAAAAAAAEAAQAAAAEAAAAAAAIAAQAAAAEAAAAAAAMAAQAAAAEAAAAAAAQAAQAAAAEAAAAAAAUAAQAAAAEAAAAAAAYAAQAAAAMAAQQJAAAAAgABAAMAAQQJAAEAAgABAAMAAQQJAAIAAgABAAMAAQQJAAMAAgABAAMAAQQJAAQAAgABAAMAAQQJAAUAAgABAAMAAQQJAAYAAgABWABYAAAAAAAAAwAAAAMAAAAcAAEAAAAAADwAAwABAAAAHAAEACAAAAAEAAQAAQAAAC7//wAAAC7////TAAEAAAAAAAABBgAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAD/gwAyAAAAAQAAAAAAAAAAAAAAAAAAAAABAAQEAAEBAQJYAAEBASH4DwD4GwHEAvgcA/gXBIwMAYuL+nz5tQXkD5j3CBLnEQACAQEBIVhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYAAABAQAADwACAQEEE/t3Dov6fAH6fAT+fPp8+nwHDosMCvm1Cvm1DAz6fBQAAAAAAAABAAAAAMmJbzEAAAAAzgTjFQAAAADOBOQpAAEAAAAAAAAADAAUAAQAAAABAAAAAgABAAAAAAAAAAAD6AAAAAAAAA==");
    return It(this, "_loadTestFont", t);
  }
  _prepareFontLoadEvent(t, e) {
    function n(L, D) {
      return L.charCodeAt(D) << 24 | L.charCodeAt(D + 1) << 16 | L.charCodeAt(D + 2) << 8 | L.charCodeAt(D + 3) & 255;
    }
    function s(L, D, N, B) {
      const $ = L.substring(0, D), V = L.substring(D + N);
      return $ + B + V;
    }
    let a, l;
    const u = this._document.createElement("canvas");
    u.width = 1, u.height = 1;
    const h = u.getContext("2d");
    let f = 0;
    function g(L, D) {
      if (++f > 30) {
        Rt("Load test font never loaded."), D();
        return;
      }
      if (h.font = "30px " + L, h.fillText(".", 0, 20), h.getImageData(0, 0, 1, 1).data[3] > 0) {
        D();
        return;
      }
      setTimeout(g.bind(null, L, D));
    }
    const y = `lt${Date.now()}${this.loadTestFontId++}`;
    let v = this._loadTestFont;
    v = s(v, 976, y.length, y);
    const x = 16, _ = 1482184792;
    let k = n(v, x);
    for (a = 0, l = y.length - 3; a < l; a += 4)
      k = k - _ + n(y, a) | 0;
    a < y.length && (k = k - _ + n(y + "XXX", a) | 0), v = s(v, x, 4, dk(k));
    const T = `url(data:font/opentype;base64,${btoa(v)});`, R = `@font-face {font-family:"${y}";src:${T}}`;
    this.insertRule(R);
    const P = this._document.createElement("div");
    P.style.visibility = "hidden", P.style.width = P.style.height = "10px", P.style.position = "absolute", P.style.top = P.style.left = "0px";
    for (const L of [t.loadedName, y]) {
      const D = this._document.createElement("span");
      D.textContent = "Hi", D.style.fontFamily = L, P.append(D);
    }
    this._document.body.append(P), g(y, () => {
      P.remove(), e.complete();
    });
  }
}
Eu = new WeakMap();
var te;
class xk {
  constructor(t, e = null, n, s) {
    b(this, te);
    this.compiledGlyphs = /* @__PURE__ */ Object.create(null), w(this, te, t), this._inspectFont = e, n && Object.assign(this, n), s && (this.charProcOperatorList = s);
  }
  createNativeFontFace() {
    var e;
    if (!this.data || this.disableFontFace)
      return null;
    let t;
    if (!this.cssFontInfo)
      t = new FontFace(this.loadedName, this.data, {});
    else {
      const n = {
        weight: this.cssFontInfo.fontWeight
      };
      this.cssFontInfo.italicAngle && (n.style = `oblique ${this.cssFontInfo.italicAngle}deg`), t = new FontFace(this.cssFontInfo.fontFamily, this.data, n);
    }
    return (e = this._inspectFont) == null || e.call(this, this), t;
  }
  createFontFaceRule() {
    var n;
    if (!this.data || this.disableFontFace)
      return null;
    const t = `url(data:${this.mimetype};base64,${b1(this.data)});`;
    let e;
    if (!this.cssFontInfo)
      e = `@font-face {font-family:"${this.loadedName}";src:${t}}`;
    else {
      let s = `font-weight: ${this.cssFontInfo.fontWeight};`;
      this.cssFontInfo.italicAngle && (s += `font-style: oblique ${this.cssFontInfo.italicAngle}deg;`), e = `@font-face {font-family:"${this.cssFontInfo.fontFamily}";${s}src:${t}}`;
    }
    return (n = this._inspectFont) == null || n.call(this, this, t), e;
  }
  getPathGenerator(t, e) {
    if (this.compiledGlyphs[e] !== void 0)
      return this.compiledGlyphs[e];
    const n = this.loadedName + "_path_" + e;
    let s;
    try {
      s = t.get(n);
    } catch (l) {
      Rt(`getPathGenerator - ignoring character: "${l}".`);
    }
    const a = new Path2D(s || "");
    return this.fontExtraProperties || t.delete(n), this.compiledGlyphs[e] = a;
  }
  get black() {
    return o(this, te).black;
  }
  get bold() {
    return o(this, te).bold;
  }
  get disableFontFace() {
    return o(this, te).disableFontFace ?? !1;
  }
  get fontExtraProperties() {
    return o(this, te).fontExtraProperties ?? !1;
  }
  get isInvalidPDFjsFont() {
    return o(this, te).isInvalidPDFjsFont;
  }
  get isType3Font() {
    return o(this, te).isType3Font;
  }
  get italic() {
    return o(this, te).italic;
  }
  get missingFile() {
    return o(this, te).missingFile;
  }
  get remeasure() {
    return o(this, te).remeasure;
  }
  get vertical() {
    return o(this, te).vertical;
  }
  get ascent() {
    return o(this, te).ascent;
  }
  get defaultWidth() {
    return o(this, te).defaultWidth;
  }
  get descent() {
    return o(this, te).descent;
  }
  get bbox() {
    return o(this, te).bbox;
  }
  get fontMatrix() {
    return o(this, te).fontMatrix;
  }
  get fallbackName() {
    return o(this, te).fallbackName;
  }
  get loadedName() {
    return o(this, te).loadedName;
  }
  get mimetype() {
    return o(this, te).mimetype;
  }
  get name() {
    return o(this, te).name;
  }
  get data() {
    return o(this, te).data;
  }
  clearData() {
    o(this, te).clearData();
  }
  get cssFontInfo() {
    return o(this, te).cssFontInfo;
  }
  get systemFontInfo() {
    return o(this, te).systemFontInfo;
  }
  get defaultVMetrics() {
    return o(this, te).defaultVMetrics;
  }
}
te = new WeakMap();
function Tk(p) {
  if (p instanceof URL)
    return p.href;
  if (typeof p == "string") {
    if (Hn)
      return p;
    const t = URL.parse(p, window.location);
    if (t)
      return t.href;
  }
  throw new Error("Invalid PDF url data: either string or URL-object is expected in the url property.");
}
function kk(p) {
  if (Hn && typeof Buffer < "u" && p instanceof Buffer)
    throw new Error("Please provide binary data as `Uint8Array`, rather than `Buffer`.");
  if (p instanceof Uint8Array && p.byteLength === p.buffer.byteLength)
    return p;
  if (typeof p == "string")
    return Fp(p);
  if (p instanceof ArrayBuffer || ArrayBuffer.isView(p) || typeof p == "object" && !isNaN(p == null ? void 0 : p.length))
    return new Uint8Array(p);
  throw new Error("Invalid PDF binary data: either TypedArray, string, or array-like object is expected in the data property.");
}
function nm(p) {
  if (typeof p != "string")
    return null;
  if (p.endsWith("/"))
    return p;
  throw new Error(`Invalid factory url: "${p}" must include trailing slash.`);
}
const iw = (p) => typeof p == "object" && Number.isInteger(p == null ? void 0 : p.num) && p.num >= 0 && Number.isInteger(p == null ? void 0 : p.gen) && p.gen >= 0, Pk = (p) => typeof p == "object" && typeof (p == null ? void 0 : p.name) == "string", a_ = mk.bind(null, iw, Pk);
var Nr, ly;
class Rk {
  constructor() {
    b(this, Nr, /* @__PURE__ */ new Map());
    b(this, ly, Promise.resolve());
  }
  postMessage(t, e) {
    const n = {
      data: structuredClone(t, e ? {
        transfer: e
      } : null)
    };
    o(this, ly).then(() => {
      for (const [s] of o(this, Nr))
        s.call(this, n);
    });
  }
  addEventListener(t, e, n = null) {
    let s = null;
    if ((n == null ? void 0 : n.signal) instanceof AbortSignal) {
      const {
        signal: a
      } = n;
      if (a.aborted) {
        Rt("LoopbackPort - cannot use an `aborted` signal.");
        return;
      }
      const l = () => this.removeEventListener(t, e);
      s = () => a.removeEventListener("abort", l), a.addEventListener("abort", l);
    }
    o(this, Nr).set(e, s);
  }
  removeEventListener(t, e) {
    const n = o(this, Nr).get(e);
    n == null || n(), o(this, Nr).delete(e);
  }
  terminate() {
    for (const [, t] of o(this, Nr))
      t == null || t();
    o(this, Nr).clear();
  }
}
Nr = new WeakMap(), ly = new WeakMap();
const im = {
  DATA: 1,
  ERROR: 2
}, He = {
  CANCEL: 1,
  CANCEL_COMPLETE: 2,
  CLOSE: 3,
  ENQUEUE: 4,
  ERROR: 5,
  PULL: 6,
  PULL_COMPLETE: 7,
  START_COMPLETE: 8
};
function OE() {
}
function Kn(p) {
  if (p instanceof ao || p instanceof $m || p instanceof PE || p instanceof Gd || p instanceof C0)
    return p;
  switch (p instanceof Error || typeof p == "object" && p !== null || ce('wrapReason: Expected "reason" to be a (possibly cloned) Error.'), p.name) {
    case "AbortException":
      return new ao(p.message);
    case "InvalidPDFException":
      return new $m(p.message);
    case "PasswordException":
      return new PE(p.message, p.code);
    case "ResponseException":
      return new Gd(p.message, p.status, p.missing);
    case "UnknownErrorException":
      return new C0(p.message, p.details);
  }
  return new C0(p.message, p.toString());
}
var _u, ns, l_, c_, u_, ym;
class Pd {
  constructor(t, e, n) {
    b(this, ns);
    b(this, _u, new AbortController());
    this.sourceName = t, this.targetName = e, this.comObj = n, this.callbackId = 1, this.streamId = 1, this.streamSinks = /* @__PURE__ */ Object.create(null), this.streamControllers = /* @__PURE__ */ Object.create(null), this.callbackCapabilities = /* @__PURE__ */ Object.create(null), this.actionHandler = /* @__PURE__ */ Object.create(null), n.addEventListener("message", E(this, ns, l_).bind(this), {
      signal: o(this, _u).signal
    });
  }
  on(t, e) {
    const n = this.actionHandler;
    if (n[t])
      throw new Error(`There is already an actionName called "${t}"`);
    n[t] = e;
  }
  send(t, e, n) {
    this.comObj.postMessage({
      sourceName: this.sourceName,
      targetName: this.targetName,
      action: t,
      data: e
    }, n);
  }
  sendWithPromise(t, e, n) {
    const s = this.callbackId++, a = Promise.withResolvers();
    this.callbackCapabilities[s] = a;
    try {
      this.comObj.postMessage({
        sourceName: this.sourceName,
        targetName: this.targetName,
        action: t,
        callbackId: s,
        data: e
      }, n);
    } catch (l) {
      a.reject(l);
    }
    return a.promise;
  }
  sendWithStream(t, e, n, s) {
    const a = this.streamId++, l = this.sourceName, u = this.targetName, h = this.comObj;
    return new ReadableStream({
      start: (f) => {
        const g = Promise.withResolvers();
        return this.streamControllers[a] = {
          controller: f,
          startCall: g,
          pullCall: null,
          cancelCall: null,
          isClosed: !1
        }, h.postMessage({
          sourceName: l,
          targetName: u,
          action: t,
          streamId: a,
          data: e,
          desiredSize: f.desiredSize
        }, s), g.promise;
      },
      pull: (f) => {
        const g = Promise.withResolvers();
        return this.streamControllers[a].pullCall = g, h.postMessage({
          sourceName: l,
          targetName: u,
          stream: He.PULL,
          streamId: a,
          desiredSize: f.desiredSize
        }), g.promise;
      },
      cancel: (f) => {
        $t(f instanceof Error, "cancel must have a valid reason");
        const g = Promise.withResolvers();
        return this.streamControllers[a].cancelCall = g, this.streamControllers[a].isClosed = !0, h.postMessage({
          sourceName: l,
          targetName: u,
          stream: He.CANCEL,
          streamId: a,
          reason: Kn(f)
        }), g.promise;
      }
    }, n);
  }
  destroy() {
    var t;
    (t = o(this, _u)) == null || t.abort(), w(this, _u, null);
  }
}
_u = new WeakMap(), ns = new WeakSet(), l_ = function({
  data: t
}) {
  if (t.targetName !== this.sourceName)
    return;
  if (t.stream) {
    E(this, ns, u_).call(this, t);
    return;
  }
  if (t.callback) {
    const n = t.callbackId, s = this.callbackCapabilities[n];
    if (!s)
      throw new Error(`Cannot resolve callback ${n}`);
    if (delete this.callbackCapabilities[n], t.callback === im.DATA)
      s.resolve(t.data);
    else if (t.callback === im.ERROR)
      s.reject(Kn(t.reason));
    else
      throw new Error("Unexpected callback case");
    return;
  }
  const e = this.actionHandler[t.action];
  if (!e)
    throw new Error(`Unknown action from worker: ${t.action}`);
  if (t.callbackId) {
    const n = this.sourceName, s = t.sourceName, a = this.comObj;
    Promise.try(e, t.data).then(function(l) {
      a.postMessage({
        sourceName: n,
        targetName: s,
        callback: im.DATA,
        callbackId: t.callbackId,
        data: l
      });
    }, function(l) {
      a.postMessage({
        sourceName: n,
        targetName: s,
        callback: im.ERROR,
        callbackId: t.callbackId,
        reason: Kn(l)
      });
    });
    return;
  }
  if (t.streamId) {
    E(this, ns, c_).call(this, t);
    return;
  }
  e(t.data);
}, c_ = function(t) {
  const e = t.streamId, n = this.sourceName, s = t.sourceName, a = this.comObj, l = this, u = this.actionHandler[t.action], h = {
    enqueue(f, g = 1, y) {
      if (this.isCancelled)
        return;
      const v = this.desiredSize;
      this.desiredSize -= g, v > 0 && this.desiredSize <= 0 && (this.sinkCapability = Promise.withResolvers(), this.ready = this.sinkCapability.promise), a.postMessage({
        sourceName: n,
        targetName: s,
        stream: He.ENQUEUE,
        streamId: e,
        chunk: f
      }, y);
    },
    close() {
      this.isCancelled || (this.isCancelled = !0, a.postMessage({
        sourceName: n,
        targetName: s,
        stream: He.CLOSE,
        streamId: e
      }), delete l.streamSinks[e]);
    },
    error(f) {
      $t(f instanceof Error, "error must have a valid reason"), !this.isCancelled && (this.isCancelled = !0, a.postMessage({
        sourceName: n,
        targetName: s,
        stream: He.ERROR,
        streamId: e,
        reason: Kn(f)
      }));
    },
    sinkCapability: Promise.withResolvers(),
    onPull: null,
    onCancel: null,
    isCancelled: !1,
    desiredSize: t.desiredSize,
    ready: null
  };
  h.sinkCapability.resolve(), h.ready = h.sinkCapability.promise, this.streamSinks[e] = h, Promise.try(u, t.data, h).then(function() {
    a.postMessage({
      sourceName: n,
      targetName: s,
      stream: He.START_COMPLETE,
      streamId: e,
      success: !0
    });
  }, function(f) {
    a.postMessage({
      sourceName: n,
      targetName: s,
      stream: He.START_COMPLETE,
      streamId: e,
      reason: Kn(f)
    });
  });
}, u_ = function(t) {
  const e = t.streamId, n = this.sourceName, s = t.sourceName, a = this.comObj, l = this.streamControllers[e], u = this.streamSinks[e];
  switch (t.stream) {
    case He.START_COMPLETE:
      t.success ? l.startCall.resolve() : l.startCall.reject(Kn(t.reason));
      break;
    case He.PULL_COMPLETE:
      t.success ? l.pullCall.resolve() : l.pullCall.reject(Kn(t.reason));
      break;
    case He.PULL:
      if (!u) {
        a.postMessage({
          sourceName: n,
          targetName: s,
          stream: He.PULL_COMPLETE,
          streamId: e,
          success: !0
        });
        break;
      }
      u.desiredSize <= 0 && t.desiredSize > 0 && u.sinkCapability.resolve(), u.desiredSize = t.desiredSize, Promise.try(u.onPull || OE).then(function() {
        a.postMessage({
          sourceName: n,
          targetName: s,
          stream: He.PULL_COMPLETE,
          streamId: e,
          success: !0
        });
      }, function(f) {
        a.postMessage({
          sourceName: n,
          targetName: s,
          stream: He.PULL_COMPLETE,
          streamId: e,
          reason: Kn(f)
        });
      });
      break;
    case He.ENQUEUE:
      if ($t(l, "enqueue should have stream controller"), l.isClosed)
        break;
      l.controller.enqueue(t.chunk);
      break;
    case He.CLOSE:
      if ($t(l, "close should have stream controller"), l.isClosed)
        break;
      l.isClosed = !0, l.controller.close(), E(this, ns, ym).call(this, l, e);
      break;
    case He.ERROR:
      $t(l, "error should have stream controller"), l.controller.error(Kn(t.reason)), E(this, ns, ym).call(this, l, e);
      break;
    case He.CANCEL_COMPLETE:
      t.success ? l.cancelCall.resolve() : l.cancelCall.reject(Kn(t.reason)), E(this, ns, ym).call(this, l, e);
      break;
    case He.CANCEL:
      if (!u)
        break;
      const h = Kn(t.reason);
      Promise.try(u.onCancel || OE, h).then(function() {
        a.postMessage({
          sourceName: n,
          targetName: s,
          stream: He.CANCEL_COMPLETE,
          streamId: e,
          success: !0
        });
      }, function(f) {
        a.postMessage({
          sourceName: n,
          targetName: s,
          stream: He.CANCEL_COMPLETE,
          streamId: e,
          reason: Kn(f)
        });
      }), u.sinkCapability.reject(h), u.isCancelled = !0, delete this.streamSinks[e];
      break;
    default:
      throw new Error("Unexpected stream case");
  }
}, ym = async function(t, e) {
  var n, s, a;
  await Promise.allSettled([(n = t.startCall) == null ? void 0 : n.promise, (s = t.pullCall) == null ? void 0 : s.promise, (a = t.cancelCall) == null ? void 0 : a.promise]), delete this.streamControllers[e];
};
var kf;
class h_ {
  constructor({
    enableHWA: t = !1
  }) {
    b(this, kf, !1);
    w(this, kf, t);
  }
  create(t, e) {
    if (t <= 0 || e <= 0)
      throw new Error("Invalid canvas size");
    const n = this._createCanvas(t, e);
    return {
      canvas: n,
      context: n.getContext("2d", {
        willReadFrequently: !o(this, kf)
      })
    };
  }
  reset(t, e, n) {
    if (!t.canvas)
      throw new Error("Canvas is not specified");
    if (e <= 0 || n <= 0)
      throw new Error("Invalid canvas size");
    t.canvas.width = e, t.canvas.height = n;
  }
  destroy(t) {
    if (!t.canvas)
      throw new Error("Canvas is not specified");
    t.canvas.width = 0, t.canvas.height = 0, t.canvas = null, t.context = null;
  }
  _createCanvas(t, e) {
    ce("Abstract method `_createCanvas` called.");
  }
}
kf = new WeakMap();
class Mk extends h_ {
  constructor({
    ownerDocument: t = globalThis.document,
    enableHWA: e = !1
  }) {
    super({
      enableHWA: e
    }), this._document = t;
  }
  _createCanvas(t, e) {
    const n = this._document.createElement("canvas");
    return n.width = t, n.height = e, n;
  }
}
class d_ {
  constructor({
    baseUrl: t = null,
    isCompressed: e = !0
  }) {
    this.baseUrl = t, this.isCompressed = e;
  }
  async fetch({
    name: t
  }) {
    if (!this.baseUrl)
      throw new Error("Ensure that the `cMapUrl` and `cMapPacked` API parameters are provided.");
    if (!t)
      throw new Error("CMap name must be specified.");
    const e = this.baseUrl + t + (this.isCompressed ? ".bcmap" : "");
    return this._fetch(e).then((n) => ({
      cMapData: n,
      isCompressed: this.isCompressed
    })).catch((n) => {
      throw new Error(`Unable to load ${this.isCompressed ? "binary " : ""}CMap at: ${e}`);
    });
  }
  async _fetch(t) {
    ce("Abstract method `_fetch` called.");
  }
}
class BE extends d_ {
  async _fetch(t) {
    const e = await Lh(t, this.isCompressed ? "arraybuffer" : "text");
    return e instanceof ArrayBuffer ? new Uint8Array(e) : Fp(e);
  }
}
class f_ {
  addFilter(t) {
    return "none";
  }
  addHCMFilter(t, e) {
    return "none";
  }
  addAlphaFilter(t) {
    return "none";
  }
  addLuminosityFilter(t) {
    return "none";
  }
  addHighlightHCMFilter(t, e, n, s, a) {
    return "none";
  }
  destroy(t = !1) {
  }
}
var vl, Cu, Or, Br, kn, wl, Al, ht, Cn, Rd, $c, vm, Vc, p_, sw, Wc, Md, Ld, rw, Dd;
class Lk extends f_ {
  constructor({
    docId: e,
    ownerDocument: n = globalThis.document
  }) {
    super();
    b(this, ht);
    b(this, vl);
    b(this, Cu);
    b(this, Or);
    b(this, Br);
    b(this, kn);
    b(this, wl);
    b(this, Al, 0);
    w(this, Br, e), w(this, kn, n);
  }
  addFilter(e) {
    if (!e)
      return "none";
    let n = o(this, ht, Cn).get(e);
    if (n)
      return n;
    const [s, a, l] = E(this, ht, vm).call(this, e), u = e.length === 1 ? s : `${s}${a}${l}`;
    if (n = o(this, ht, Cn).get(u), n)
      return o(this, ht, Cn).set(e, n), n;
    const h = `g_${o(this, Br)}_transfer_map_${Ge(this, Al)._++}`, f = E(this, ht, Vc).call(this, h);
    o(this, ht, Cn).set(e, f), o(this, ht, Cn).set(u, f);
    const g = E(this, ht, Wc).call(this, h);
    return E(this, ht, Ld).call(this, s, a, l, g), f;
  }
  addHCMFilter(e, n) {
    var x;
    const s = `${e}-${n}`, a = "base";
    let l = o(this, ht, Rd).get(a);
    if ((l == null ? void 0 : l.key) === s || (l ? ((x = l.filter) == null || x.remove(), l.key = s, l.url = "none", l.filter = null) : (l = {
      key: s,
      url: "none",
      filter: null
    }, o(this, ht, Rd).set(a, l)), !e || !n))
      return l.url;
    const u = E(this, ht, Dd).call(this, e);
    e = ft.makeHexColor(...u);
    const h = E(this, ht, Dd).call(this, n);
    if (n = ft.makeHexColor(...h), o(this, ht, $c).style.color = "", e === "#000000" && n === "#ffffff" || e === n)
      return l.url;
    const f = new Array(256);
    for (let _ = 0; _ <= 255; _++) {
      const k = _ / 255;
      f[_] = k <= 0.03928 ? k / 12.92 : ((k + 0.055) / 1.055) ** 2.4;
    }
    const g = f.join(","), y = `g_${o(this, Br)}_hcm_filter`, v = l.filter = E(this, ht, Wc).call(this, y);
    E(this, ht, Ld).call(this, g, g, g, v), E(this, ht, sw).call(this, v);
    const S = (_, k) => {
      const T = u[_] / 255, R = h[_] / 255, P = new Array(k + 1);
      for (let L = 0; L <= k; L++)
        P[L] = T + L / k * (R - T);
      return P.join(",");
    };
    return E(this, ht, Ld).call(this, S(0, 5), S(1, 5), S(2, 5), v), l.url = E(this, ht, Vc).call(this, y), l.url;
  }
  addAlphaFilter(e) {
    let n = o(this, ht, Cn).get(e);
    if (n)
      return n;
    const [s] = E(this, ht, vm).call(this, [e]), a = `alpha_${s}`;
    if (n = o(this, ht, Cn).get(a), n)
      return o(this, ht, Cn).set(e, n), n;
    const l = `g_${o(this, Br)}_alpha_map_${Ge(this, Al)._++}`, u = E(this, ht, Vc).call(this, l);
    o(this, ht, Cn).set(e, u), o(this, ht, Cn).set(a, u);
    const h = E(this, ht, Wc).call(this, l);
    return E(this, ht, rw).call(this, s, h), u;
  }
  addLuminosityFilter(e) {
    let n = o(this, ht, Cn).get(e || "luminosity");
    if (n)
      return n;
    let s, a;
    if (e ? ([s] = E(this, ht, vm).call(this, [e]), a = `luminosity_${s}`) : a = "luminosity", n = o(this, ht, Cn).get(a), n)
      return o(this, ht, Cn).set(e, n), n;
    const l = `g_${o(this, Br)}_luminosity_map_${Ge(this, Al)._++}`, u = E(this, ht, Vc).call(this, l);
    o(this, ht, Cn).set(e, u), o(this, ht, Cn).set(a, u);
    const h = E(this, ht, Wc).call(this, l);
    return E(this, ht, p_).call(this, h), e && E(this, ht, rw).call(this, s, h), u;
  }
  addHighlightHCMFilter(e, n, s, a, l) {
    var R;
    const u = `${n}-${s}-${a}-${l}`;
    let h = o(this, ht, Rd).get(e);
    if ((h == null ? void 0 : h.key) === u || (h ? ((R = h.filter) == null || R.remove(), h.key = u, h.url = "none", h.filter = null) : (h = {
      key: u,
      url: "none",
      filter: null
    }, o(this, ht, Rd).set(e, h)), !n || !s))
      return h.url;
    const [f, g] = [n, s].map(E(this, ht, Dd).bind(this));
    let y = Math.round(0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]), v = Math.round(0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]), [S, x] = [a, l].map(E(this, ht, Dd).bind(this));
    v < y && ([y, v, S, x] = [v, y, x, S]), o(this, ht, $c).style.color = "";
    const _ = (P, L, D) => {
      const N = new Array(256), B = (v - y) / D, $ = P / 255, V = (L - P) / (255 * D);
      let X = 0;
      for (let q = 0; q <= D; q++) {
        const it = Math.round(y + q * B), lt = $ + q * V;
        for (let rt = X; rt <= it; rt++)
          N[rt] = lt;
        X = it + 1;
      }
      for (let q = X; q < 256; q++)
        N[q] = N[X - 1];
      return N.join(",");
    }, k = `g_${o(this, Br)}_hcm_${e}_filter`, T = h.filter = E(this, ht, Wc).call(this, k);
    return E(this, ht, sw).call(this, T), E(this, ht, Ld).call(this, _(S[0], x[0], 5), _(S[1], x[1], 5), _(S[2], x[2], 5), T), h.url = E(this, ht, Vc).call(this, k), h.url;
  }
  destroy(e = !1) {
    var n, s, a, l;
    e && ((n = o(this, wl)) != null && n.size) || ((s = o(this, Or)) == null || s.parentNode.parentNode.remove(), w(this, Or, null), (a = o(this, Cu)) == null || a.clear(), w(this, Cu, null), (l = o(this, wl)) == null || l.clear(), w(this, wl, null), w(this, Al, 0));
  }
}
vl = new WeakMap(), Cu = new WeakMap(), Or = new WeakMap(), Br = new WeakMap(), kn = new WeakMap(), wl = new WeakMap(), Al = new WeakMap(), ht = new WeakSet(), Cn = function() {
  return o(this, Cu) || w(this, Cu, /* @__PURE__ */ new Map());
}, Rd = function() {
  return o(this, wl) || w(this, wl, /* @__PURE__ */ new Map());
}, $c = function() {
  if (!o(this, Or)) {
    const e = o(this, kn).createElement("div"), {
      style: n
    } = e;
    n.visibility = "hidden", n.contain = "strict", n.width = n.height = 0, n.position = "absolute", n.top = n.left = 0, n.zIndex = -1;
    const s = o(this, kn).createElementNS(Ar, "svg");
    s.setAttribute("width", 0), s.setAttribute("height", 0), w(this, Or, o(this, kn).createElementNS(Ar, "defs")), e.append(s), s.append(o(this, Or)), o(this, kn).body.append(e);
  }
  return o(this, Or);
}, vm = function(e) {
  if (e.length === 1) {
    const f = e[0], g = new Array(256);
    for (let v = 0; v < 256; v++)
      g[v] = f[v] / 255;
    const y = g.join(",");
    return [y, y, y];
  }
  const [n, s, a] = e, l = new Array(256), u = new Array(256), h = new Array(256);
  for (let f = 0; f < 256; f++)
    l[f] = n[f] / 255, u[f] = s[f] / 255, h[f] = a[f] / 255;
  return [l.join(","), u.join(","), h.join(",")];
}, Vc = function(e) {
  if (o(this, vl) === void 0) {
    w(this, vl, "");
    const n = o(this, kn).URL;
    n !== o(this, kn).baseURI && (Op(n) ? Rt('#createUrl: ignore "data:"-URL for performance reasons.') : w(this, vl, cA(n, "")));
  }
  return `url(${o(this, vl)}#${e})`;
}, p_ = function(e) {
  const n = o(this, kn).createElementNS(Ar, "feColorMatrix");
  n.setAttribute("type", "matrix"), n.setAttribute("values", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0.59 0.11 0 0"), e.append(n);
}, sw = function(e) {
  const n = o(this, kn).createElementNS(Ar, "feColorMatrix");
  n.setAttribute("type", "matrix"), n.setAttribute("values", "0.2126 0.7152 0.0722 0 0 0.2126 0.7152 0.0722 0 0 0.2126 0.7152 0.0722 0 0 0 0 0 1 0"), e.append(n);
}, Wc = function(e) {
  const n = o(this, kn).createElementNS(Ar, "filter");
  return n.setAttribute("color-interpolation-filters", "sRGB"), n.setAttribute("id", e), o(this, ht, $c).append(n), n;
}, Md = function(e, n, s) {
  const a = o(this, kn).createElementNS(Ar, n);
  a.setAttribute("type", "discrete"), a.setAttribute("tableValues", s), e.append(a);
}, Ld = function(e, n, s, a) {
  const l = o(this, kn).createElementNS(Ar, "feComponentTransfer");
  a.append(l), E(this, ht, Md).call(this, l, "feFuncR", e), E(this, ht, Md).call(this, l, "feFuncG", n), E(this, ht, Md).call(this, l, "feFuncB", s);
}, rw = function(e, n) {
  const s = o(this, kn).createElementNS(Ar, "feComponentTransfer");
  n.append(s), E(this, ht, Md).call(this, s, "feFuncA", e);
}, Dd = function(e) {
  return o(this, ht, $c).style.color = e, Dh(getComputedStyle(o(this, ht, $c)).getPropertyValue("color"));
};
class g_ {
  constructor({
    baseUrl: t = null
  }) {
    this.baseUrl = t;
  }
  async fetch({
    filename: t
  }) {
    if (!this.baseUrl)
      throw new Error("Ensure that the `standardFontDataUrl` API parameter is provided.");
    if (!t)
      throw new Error("Font filename must be specified.");
    const e = `${this.baseUrl}${t}`;
    return this._fetch(e).catch((n) => {
      throw new Error(`Unable to load font data at: ${e}`);
    });
  }
  async _fetch(t) {
    ce("Abstract method `_fetch` called.");
  }
}
class zE extends g_ {
  async _fetch(t) {
    const e = await Lh(t, "arraybuffer");
    return new Uint8Array(e);
  }
}
class m_ {
  constructor({
    baseUrl: t = null
  }) {
    this.baseUrl = t;
  }
  async fetch({
    filename: t
  }) {
    if (!this.baseUrl)
      throw new Error("Ensure that the `wasmUrl` API parameter is provided.");
    if (!t)
      throw new Error("Wasm filename must be specified.");
    const e = `${this.baseUrl}${t}`;
    return this._fetch(e).catch((n) => {
      throw new Error(`Unable to load wasm data at: ${e}`);
    });
  }
  async _fetch(t) {
    ce("Abstract method `_fetch` called.");
  }
}
class UE extends m_ {
  async _fetch(t) {
    const e = await Lh(t, "arraybuffer");
    return new Uint8Array(e);
  }
}
Hn && Rt("Please use the `legacy` build in Node.js environments.");
async function gA(p) {
  const e = await process.getBuiltinModule("fs").promises.readFile(p);
  return new Uint8Array(e);
}
class Dk extends f_ {
}
class Ik extends h_ {
  _createCanvas(t, e) {
    return process.getBuiltinModule("module").createRequire(import.meta.url)("@napi-rs/canvas").createCanvas(t, e);
  }
}
class Fk extends d_ {
  async _fetch(t) {
    return gA(t);
  }
}
class Nk extends g_ {
  async _fetch(t) {
    return gA(t);
  }
}
class Ok extends m_ {
  async _fetch(t) {
    return gA(t);
  }
}
const Uc = "__forcedDependency", {
  floor: HE,
  ceil: jE
} = Math;
function sm(p, t, e, n, s, a) {
  p[t * 4 + 0] = Math.min(p[t * 4 + 0], e), p[t * 4 + 1] = Math.min(p[t * 4 + 1], n), p[t * 4 + 2] = Math.max(p[t * 4 + 2], s), p[t * 4 + 3] = Math.max(p[t * 4 + 3], a);
}
const ow = new Uint32Array(new Uint8Array([255, 255, 0, 0]).buffer)[0];
var xu, Qo;
class Bk {
  constructor(t, e) {
    b(this, xu);
    b(this, Qo);
    w(this, xu, t), w(this, Qo, e);
  }
  get length() {
    return o(this, xu).length;
  }
  isEmpty(t) {
    return o(this, xu)[t] === ow;
  }
  minX(t) {
    return o(this, Qo)[t * 4 + 0] / 256;
  }
  minY(t) {
    return o(this, Qo)[t * 4 + 1] / 256;
  }
  maxX(t) {
    return (o(this, Qo)[t * 4 + 2] + 1) / 256;
  }
  maxY(t) {
    return (o(this, Qo)[t * 4 + 3] + 1) / 256;
  }
}
xu = new WeakMap(), Qo = new WeakMap();
const rm = (p, t) => {
  if (!p)
    return;
  let e = p.get(t);
  return e || (e = {
    dependencies: /* @__PURE__ */ new Set(),
    isRenderingOperation: !1
  }, p.set(t, e)), e;
};
var Jn, ti, Sl, vs, bl, zr, Zt, ee, Ur, ei, Pf, Tu, El, _l, Hr, Pn, Ys, Rf, aw;
class zk {
  constructor(t, e, n = !1) {
    b(this, Rf);
    b(this, Jn, {
      __proto__: null
    });
    b(this, ti, {
      __proto__: null,
      transform: [],
      moveText: [],
      sameLineText: [],
      [Uc]: []
    });
    b(this, Sl, /* @__PURE__ */ new Map());
    b(this, vs, []);
    b(this, bl, []);
    b(this, zr, [[1, 0, 0, 1, 0, 0]]);
    b(this, Zt, [-1 / 0, -1 / 0, 1 / 0, 1 / 0]);
    b(this, ee, new Float64Array([1 / 0, 1 / 0, -1 / 0, -1 / 0]));
    b(this, Ur, -1);
    b(this, ei, /* @__PURE__ */ new Set());
    b(this, Pf, /* @__PURE__ */ new Map());
    b(this, Tu, /* @__PURE__ */ new Map());
    b(this, El);
    b(this, _l);
    b(this, Hr);
    b(this, Pn);
    b(this, Ys);
    w(this, El, t.width), w(this, _l, t.height), E(this, Rf, aw).call(this, e), n && w(this, Ys, /* @__PURE__ */ new Map());
  }
  growOperationsCount(t) {
    t >= o(this, Pn).length && E(this, Rf, aw).call(this, t, o(this, Pn));
  }
  save(t) {
    return w(this, Jn, {
      __proto__: o(this, Jn)
    }), w(this, ti, {
      __proto__: o(this, ti),
      transform: {
        __proto__: o(this, ti).transform
      },
      moveText: {
        __proto__: o(this, ti).moveText
      },
      sameLineText: {
        __proto__: o(this, ti).sameLineText
      },
      [Uc]: {
        __proto__: o(this, ti)[Uc]
      }
    }), w(this, Zt, {
      __proto__: o(this, Zt)
    }), o(this, vs).push(t), this;
  }
  restore(t) {
    var s;
    const e = Object.getPrototypeOf(o(this, Jn));
    if (e === null)
      return this;
    w(this, Jn, e), w(this, ti, Object.getPrototypeOf(o(this, ti))), w(this, Zt, Object.getPrototypeOf(o(this, Zt)));
    const n = o(this, vs).pop();
    return n !== void 0 && ((s = rm(o(this, Ys), t)) == null || s.dependencies.add(n), o(this, Pn)[t] = o(this, Pn)[n]), this;
  }
  recordOpenMarker(t) {
    return o(this, vs).push(t), this;
  }
  getOpenMarker() {
    return o(this, vs).length === 0 ? null : o(this, vs).at(-1);
  }
  recordCloseMarker(t) {
    var n;
    const e = o(this, vs).pop();
    return e !== void 0 && ((n = rm(o(this, Ys), t)) == null || n.dependencies.add(e), o(this, Pn)[t] = o(this, Pn)[e]), this;
  }
  beginMarkedContent(t) {
    return o(this, bl).push(t), this;
  }
  endMarkedContent(t) {
    var n;
    const e = o(this, bl).pop();
    return e !== void 0 && ((n = rm(o(this, Ys), t)) == null || n.dependencies.add(e), o(this, Pn)[t] = o(this, Pn)[e]), this;
  }
  pushBaseTransform(t) {
    return o(this, zr).push(ft.multiplyByDOMMatrix(o(this, zr).at(-1), t.getTransform())), this;
  }
  popBaseTransform() {
    return o(this, zr).length > 1 && o(this, zr).pop(), this;
  }
  recordSimpleData(t, e) {
    return o(this, Jn)[t] = e, this;
  }
  recordIncrementalData(t, e) {
    return o(this, ti)[t].push(e), this;
  }
  resetIncrementalData(t, e) {
    return o(this, ti)[t].length = 0, this;
  }
  recordNamedData(t, e) {
    return o(this, Sl).set(t, e), this;
  }
  recordSimpleDataFromNamed(t, e, n) {
    o(this, Jn)[t] = o(this, Sl).get(e) ?? n;
  }
  recordFutureForcedDependency(t, e) {
    return this.recordIncrementalData(Uc, e), this;
  }
  inheritSimpleDataAsFutureForcedDependencies(t) {
    for (const e of t)
      e in o(this, Jn) && this.recordFutureForcedDependency(e, o(this, Jn)[e]);
    return this;
  }
  inheritPendingDependenciesAsFutureForcedDependencies() {
    for (const t of o(this, ei))
      this.recordFutureForcedDependency(Uc, t);
    return this;
  }
  resetBBox(t) {
    return o(this, Ur) !== t && (w(this, Ur, t), o(this, ee)[0] = 1 / 0, o(this, ee)[1] = 1 / 0, o(this, ee)[2] = -1 / 0, o(this, ee)[3] = -1 / 0), this;
  }
  recordClipBox(t, e, n, s, a, l) {
    const u = ft.multiplyByDOMMatrix(o(this, zr).at(-1), e.getTransform()), h = [1 / 0, 1 / 0, -1 / 0, -1 / 0];
    ft.axialAlignedBoundingBox([n, a, s, l], u, h);
    const f = ft.intersect(o(this, Zt), h);
    return f ? (o(this, Zt)[0] = f[0], o(this, Zt)[1] = f[1], o(this, Zt)[2] = f[2], o(this, Zt)[3] = f[3]) : (o(this, Zt)[0] = o(this, Zt)[1] = 1 / 0, o(this, Zt)[2] = o(this, Zt)[3] = -1 / 0), this;
  }
  recordBBox(t, e, n, s, a, l) {
    const u = o(this, Zt);
    if (u[0] === 1 / 0)
      return this;
    const h = ft.multiplyByDOMMatrix(o(this, zr).at(-1), e.getTransform());
    if (u[0] === -1 / 0)
      return ft.axialAlignedBoundingBox([n, a, s, l], h, o(this, ee)), this;
    const f = [1 / 0, 1 / 0, -1 / 0, -1 / 0];
    return ft.axialAlignedBoundingBox([n, a, s, l], h, f), o(this, ee)[0] = Math.min(o(this, ee)[0], Math.max(f[0], u[0])), o(this, ee)[1] = Math.min(o(this, ee)[1], Math.max(f[1], u[1])), o(this, ee)[2] = Math.max(o(this, ee)[2], Math.min(f[2], u[2])), o(this, ee)[3] = Math.max(o(this, ee)[3], Math.min(f[3], u[3])), this;
  }
  recordCharacterBBox(t, e, n, s = 1, a = 0, l = 0, u) {
    const h = n.bbox;
    let f, g;
    if (h && (f = h[2] !== h[0] && h[3] !== h[1] && o(this, Tu).get(n), f !== !1 && (g = [0, 0, 0, 0], ft.axialAlignedBoundingBox(h, n.fontMatrix, g), (s !== 1 || a !== 0 || l !== 0) && ft.scaleMinMax([s, 0, 0, -s, a, l], g), f)))
      return this.recordBBox(t, e, g[0], g[2], g[1], g[3]);
    if (!u)
      return this.recordFullPageBBox(t);
    const y = u();
    return h && g && f === void 0 && (f = g[0] <= a - y.actualBoundingBoxLeft && g[2] >= a + y.actualBoundingBoxRight && g[1] <= l - y.actualBoundingBoxAscent && g[3] >= l + y.actualBoundingBoxDescent, o(this, Tu).set(n, f), f) ? this.recordBBox(t, e, g[0], g[2], g[1], g[3]) : this.recordBBox(t, e, a - y.actualBoundingBoxLeft, a + y.actualBoundingBoxRight, l - y.actualBoundingBoxAscent, l + y.actualBoundingBoxDescent);
  }
  recordFullPageBBox(t) {
    return o(this, ee)[0] = Math.max(0, o(this, Zt)[0]), o(this, ee)[1] = Math.max(0, o(this, Zt)[1]), o(this, ee)[2] = Math.min(o(this, El), o(this, Zt)[2]), o(this, ee)[3] = Math.min(o(this, _l), o(this, Zt)[3]), this;
  }
  getSimpleIndex(t) {
    return o(this, Jn)[t];
  }
  recordDependencies(t, e) {
    const n = o(this, ei), s = o(this, Jn), a = o(this, ti);
    for (const l of e)
      l in o(this, Jn) ? n.add(s[l]) : l in a && a[l].forEach(n.add, n);
    return this;
  }
  recordNamedDependency(t, e) {
    return o(this, Sl).has(e) && o(this, ei).add(o(this, Sl).get(e)), this;
  }
  recordOperation(t, e = !1) {
    if (this.recordDependencies(t, [Uc]), o(this, Ys)) {
      const n = rm(o(this, Ys), t), {
        dependencies: s
      } = n;
      o(this, ei).forEach(s.add, s), o(this, vs).forEach(s.add, s), o(this, bl).forEach(s.add, s), s.delete(t), n.isRenderingOperation = !0;
    }
    if (o(this, Ur) === t) {
      const n = HE(o(this, ee)[0] * 256 / o(this, El)), s = HE(o(this, ee)[1] * 256 / o(this, _l)), a = jE(o(this, ee)[2] * 256 / o(this, El)), l = jE(o(this, ee)[3] * 256 / o(this, _l));
      sm(o(this, Hr), t, n, s, a, l);
      for (const u of o(this, ei))
        u !== t && sm(o(this, Hr), u, n, s, a, l);
      for (const u of o(this, vs))
        u !== t && sm(o(this, Hr), u, n, s, a, l);
      for (const u of o(this, bl))
        u !== t && sm(o(this, Hr), u, n, s, a, l);
      e || (o(this, ei).clear(), w(this, Ur, -1));
    }
    return this;
  }
  recordShowTextOperation(t, e = !1) {
    const n = Array.from(o(this, ei));
    this.recordOperation(t, e), this.recordIncrementalData("sameLineText", t);
    for (const s of n)
      this.recordIncrementalData("sameLineText", s);
    return this;
  }
  bboxToClipBoxDropOperation(t, e = !1) {
    return o(this, Ur) === t && (w(this, Ur, -1), o(this, Zt)[0] = Math.max(o(this, Zt)[0], o(this, ee)[0]), o(this, Zt)[1] = Math.max(o(this, Zt)[1], o(this, ee)[1]), o(this, Zt)[2] = Math.min(o(this, Zt)[2], o(this, ee)[2]), o(this, Zt)[3] = Math.min(o(this, Zt)[3], o(this, ee)[3]), e || o(this, ei).clear()), this;
  }
  _takePendingDependencies() {
    const t = o(this, ei);
    return w(this, ei, /* @__PURE__ */ new Set()), t;
  }
  _extractOperation(t) {
    const e = o(this, Pf).get(t);
    return o(this, Pf).delete(t), e;
  }
  _pushPendingDependencies(t) {
    for (const e of t)
      o(this, ei).add(e);
  }
  take() {
    return o(this, Tu).clear(), new Bk(o(this, Pn), o(this, Hr));
  }
  takeDebugMetadata() {
    return o(this, Ys);
  }
}
Jn = new WeakMap(), ti = new WeakMap(), Sl = new WeakMap(), vs = new WeakMap(), bl = new WeakMap(), zr = new WeakMap(), Zt = new WeakMap(), ee = new WeakMap(), Ur = new WeakMap(), ei = new WeakMap(), Pf = new WeakMap(), Tu = new WeakMap(), El = new WeakMap(), _l = new WeakMap(), Hr = new WeakMap(), Pn = new WeakMap(), Ys = new WeakMap(), Rf = new WeakSet(), aw = function(t, e) {
  const n = new ArrayBuffer(t * 4);
  w(this, Hr, new Uint8ClampedArray(n)), w(this, Pn, new Uint32Array(n)), e && e.length > 0 ? (o(this, Pn).set(e), o(this, Pn).fill(ow, e.length)) : o(this, Pn).fill(ow);
};
var oe, Re, ws, ku, Pu;
const xA = class xA {
  constructor(t, e, n) {
    b(this, oe);
    b(this, Re);
    b(this, ws);
    b(this, ku, 0);
    b(this, Pu, 0);
    if (t instanceof xA && o(t, ws) === !!n)
      return t;
    w(this, oe, t), w(this, Re, e), w(this, ws, !!n);
  }
  growOperationsCount() {
    throw new Error("Unreachable");
  }
  save(t) {
    return Ge(this, Pu)._++, o(this, oe).save(o(this, Re)), this;
  }
  restore(t) {
    return o(this, Pu) > 0 && (o(this, oe).restore(o(this, Re)), Ge(this, Pu)._--), this;
  }
  recordOpenMarker(t) {
    return Ge(this, ku)._++, this;
  }
  getOpenMarker() {
    return o(this, ku) > 0 ? o(this, Re) : o(this, oe).getOpenMarker();
  }
  recordCloseMarker(t) {
    return Ge(this, ku)._--, this;
  }
  beginMarkedContent(t) {
    return this;
  }
  endMarkedContent(t) {
    return this;
  }
  pushBaseTransform(t) {
    return o(this, oe).pushBaseTransform(t), this;
  }
  popBaseTransform() {
    return o(this, oe).popBaseTransform(), this;
  }
  recordSimpleData(t, e) {
    return o(this, oe).recordSimpleData(t, o(this, Re)), this;
  }
  recordIncrementalData(t, e) {
    return o(this, oe).recordIncrementalData(t, o(this, Re)), this;
  }
  resetIncrementalData(t, e) {
    return o(this, oe).resetIncrementalData(t, o(this, Re)), this;
  }
  recordNamedData(t, e) {
    return this;
  }
  recordSimpleDataFromNamed(t, e, n) {
    return o(this, oe).recordSimpleDataFromNamed(t, e, o(this, Re)), this;
  }
  recordFutureForcedDependency(t, e) {
    return o(this, oe).recordFutureForcedDependency(t, o(this, Re)), this;
  }
  inheritSimpleDataAsFutureForcedDependencies(t) {
    return o(this, oe).inheritSimpleDataAsFutureForcedDependencies(t), this;
  }
  inheritPendingDependenciesAsFutureForcedDependencies() {
    return o(this, oe).inheritPendingDependenciesAsFutureForcedDependencies(), this;
  }
  resetBBox(t) {
    return o(this, ws) || o(this, oe).resetBBox(o(this, Re)), this;
  }
  recordClipBox(t, e, n, s, a, l) {
    return o(this, ws) || o(this, oe).recordClipBox(o(this, Re), e, n, s, a, l), this;
  }
  recordBBox(t, e, n, s, a, l) {
    return o(this, ws) || o(this, oe).recordBBox(o(this, Re), e, n, s, a, l), this;
  }
  recordCharacterBBox(t, e, n, s, a, l, u) {
    return o(this, ws) || o(this, oe).recordCharacterBBox(o(this, Re), e, n, s, a, l, u), this;
  }
  recordFullPageBBox(t) {
    return o(this, ws) || o(this, oe).recordFullPageBBox(o(this, Re)), this;
  }
  getSimpleIndex(t) {
    return o(this, oe).getSimpleIndex(t);
  }
  recordDependencies(t, e) {
    return o(this, oe).recordDependencies(o(this, Re), e), this;
  }
  recordNamedDependency(t, e) {
    return o(this, oe).recordNamedDependency(o(this, Re), e), this;
  }
  recordOperation(t) {
    return o(this, oe).recordOperation(o(this, Re), !0), this;
  }
  recordShowTextOperation(t) {
    return o(this, oe).recordShowTextOperation(o(this, Re), !0), this;
  }
  bboxToClipBoxDropOperation(t) {
    return o(this, ws) || o(this, oe).bboxToClipBoxDropOperation(o(this, Re), !0), this;
  }
  take() {
    throw new Error("Unreachable");
  }
  takeDebugMetadata() {
    throw new Error("Unreachable");
  }
};
oe = new WeakMap(), Re = new WeakMap(), ws = new WeakMap(), ku = new WeakMap(), Pu = new WeakMap();
let Gm = xA;
const ji = {
  stroke: ["path", "transform", "filter", "strokeColor", "strokeAlpha", "lineWidth", "lineCap", "lineJoin", "miterLimit", "dash"],
  fill: ["path", "transform", "filter", "fillColor", "fillAlpha", "globalCompositeOperation", "SMask"],
  imageXObject: ["transform", "SMask", "filter", "fillAlpha", "strokeAlpha", "globalCompositeOperation"],
  rawFillPath: ["filter", "fillColor", "fillAlpha"],
  showText: ["transform", "leading", "charSpacing", "wordSpacing", "hScale", "textRise", "moveText", "textMatrix", "font", "fontObj", "filter", "fillColor", "textRenderingMode", "SMask", "fillAlpha", "strokeAlpha", "globalCompositeOperation", "sameLineText"],
  transform: ["transform"],
  transformAndFill: ["transform", "fillColor"]
}, vn = {
  FILL: "Fill",
  STROKE: "Stroke",
  SHADING: "Shading"
};
function lw(p, t) {
  if (!t)
    return;
  const e = t[2] - t[0], n = t[3] - t[1], s = new Path2D();
  s.rect(t[0], t[1], e, n), p.clip(s);
}
class mA {
  isModifyingCurrentTransform() {
    return !1;
  }
  getPattern() {
    ce("Abstract method `getPattern` called.");
  }
}
class Uk extends mA {
  constructor(t) {
    super(), this._type = t[1], this._bbox = t[2], this._colorStops = t[3], this._p0 = t[4], this._p1 = t[5], this._r0 = t[6], this._r1 = t[7], this.matrix = null;
  }
  _createGradient(t) {
    let e;
    this._type === "axial" ? e = t.createLinearGradient(this._p0[0], this._p0[1], this._p1[0], this._p1[1]) : this._type === "radial" && (e = t.createRadialGradient(this._p0[0], this._p0[1], this._r0, this._p1[0], this._p1[1], this._r1));
    for (const n of this._colorStops)
      e.addColorStop(n[0], n[1]);
    return e;
  }
  getPattern(t, e, n, s) {
    let a;
    if (s === vn.STROKE || s === vn.FILL) {
      const l = e.current.getClippedPathBoundingBox(s, _e(t)) || [0, 0, 0, 0], u = Math.ceil(l[2] - l[0]) || 1, h = Math.ceil(l[3] - l[1]) || 1, f = e.cachedCanvases.getCanvas("pattern", u, h), g = f.context;
      g.clearRect(0, 0, g.canvas.width, g.canvas.height), g.beginPath(), g.rect(0, 0, g.canvas.width, g.canvas.height), g.translate(-l[0], -l[1]), n = ft.transform(n, [1, 0, 0, 1, l[0], l[1]]), g.transform(...e.baseTransform), this.matrix && g.transform(...this.matrix), lw(g, this._bbox), g.fillStyle = this._createGradient(g), g.fill(), a = t.createPattern(f.canvas, "no-repeat");
      const y = new DOMMatrix(n);
      a.setTransform(y);
    } else
      lw(t, this._bbox), a = this._createGradient(t);
    return a;
  }
}
function k0(p, t, e, n, s, a, l, u) {
  const h = t.coords, f = t.colors, g = p.data, y = p.width * 4;
  let v;
  h[e + 1] > h[n + 1] && (v = e, e = n, n = v, v = a, a = l, l = v), h[n + 1] > h[s + 1] && (v = n, n = s, s = v, v = l, l = u, u = v), h[e + 1] > h[n + 1] && (v = e, e = n, n = v, v = a, a = l, l = v);
  const S = (h[e] + t.offsetX) * t.scaleX, x = (h[e + 1] + t.offsetY) * t.scaleY, _ = (h[n] + t.offsetX) * t.scaleX, k = (h[n + 1] + t.offsetY) * t.scaleY, T = (h[s] + t.offsetX) * t.scaleX, R = (h[s + 1] + t.offsetY) * t.scaleY;
  if (x >= R)
    return;
  const P = f[a], L = f[a + 1], D = f[a + 2], N = f[l], B = f[l + 1], $ = f[l + 2], V = f[u], X = f[u + 1], q = f[u + 2], it = Math.round(x), lt = Math.round(R);
  let rt, vt, tt, Q, W, ot, st, I;
  for (let z = it; z <= lt; z++) {
    if (z < k) {
      const bt = z < x ? 0 : (x - z) / (x - k);
      rt = S - (S - _) * bt, vt = P - (P - N) * bt, tt = L - (L - B) * bt, Q = D - (D - $) * bt;
    } else {
      let bt;
      z > R ? bt = 1 : k === R ? bt = 0 : bt = (k - z) / (k - R), rt = _ - (_ - T) * bt, vt = N - (N - V) * bt, tt = B - (B - X) * bt, Q = $ - ($ - q) * bt;
    }
    let K;
    z < x ? K = 0 : z > R ? K = 1 : K = (x - z) / (x - R), W = S - (S - T) * K, ot = P - (P - V) * K, st = L - (L - X) * K, I = D - (D - q) * K;
    const ct = Math.round(Math.min(rt, W)), Et = Math.round(Math.max(rt, W));
    let xt = y * z + ct * 4;
    for (let bt = ct; bt <= Et; bt++)
      K = (rt - bt) / (rt - W), K < 0 ? K = 0 : K > 1 && (K = 1), g[xt++] = vt - (vt - ot) * K | 0, g[xt++] = tt - (tt - st) * K | 0, g[xt++] = Q - (Q - I) * K | 0, g[xt++] = 255;
  }
}
function Hk(p, t, e) {
  const n = t.coords, s = t.colors;
  let a, l;
  switch (t.type) {
    case "lattice":
      const u = t.verticesPerRow, h = Math.floor(n.length / u) - 1, f = u - 1;
      for (a = 0; a < h; a++) {
        let g = a * u;
        for (let y = 0; y < f; y++, g++)
          k0(p, e, n[g], n[g + 1], n[g + u], s[g], s[g + 1], s[g + u]), k0(p, e, n[g + u + 1], n[g + 1], n[g + u], s[g + u + 1], s[g + 1], s[g + u]);
      }
      break;
    case "triangles":
      for (a = 0, l = n.length; a < l; a += 3)
        k0(p, e, n[a], n[a + 1], n[a + 2], s[a], s[a + 1], s[a + 2]);
      break;
    default:
      throw new Error("illegal figure");
  }
}
class jk extends mA {
  constructor(t) {
    super(), this._coords = t[2], this._colors = t[3], this._figures = t[4], this._bounds = t[5], this._bbox = t[6], this._background = t[7], this.matrix = null;
  }
  _createMeshCanvas(t, e, n) {
    const u = Math.floor(this._bounds[0]), h = Math.floor(this._bounds[1]), f = Math.ceil(this._bounds[2]) - u, g = Math.ceil(this._bounds[3]) - h, y = Math.min(Math.ceil(Math.abs(f * t[0] * 1.1)), 3e3), v = Math.min(Math.ceil(Math.abs(g * t[1] * 1.1)), 3e3), S = f / y, x = g / v, _ = {
      coords: this._coords,
      colors: this._colors,
      offsetX: -u,
      offsetY: -h,
      scaleX: 1 / S,
      scaleY: 1 / x
    }, k = y + 2 * 2, T = v + 2 * 2, R = n.getCanvas("mesh", k, T), P = R.context, L = P.createImageData(y, v);
    if (e) {
      const N = L.data;
      for (let B = 0, $ = N.length; B < $; B += 4)
        N[B] = e[0], N[B + 1] = e[1], N[B + 2] = e[2], N[B + 3] = 255;
    }
    for (const N of this._figures)
      Hk(L, N, _);
    return P.putImageData(L, 2, 2), {
      canvas: R.canvas,
      offsetX: u - 2 * S,
      offsetY: h - 2 * x,
      scaleX: S,
      scaleY: x
    };
  }
  isModifyingCurrentTransform() {
    return !0;
  }
  getPattern(t, e, n, s) {
    lw(t, this._bbox);
    const a = new Float32Array(2);
    if (s === vn.SHADING)
      ft.singularValueDecompose2dScale(_e(t), a);
    else if (this.matrix) {
      ft.singularValueDecompose2dScale(this.matrix, a);
      const [u, h] = a;
      ft.singularValueDecompose2dScale(e.baseTransform, a), a[0] *= u, a[1] *= h;
    } else
      ft.singularValueDecompose2dScale(e.baseTransform, a);
    const l = this._createMeshCanvas(a, s === vn.SHADING ? null : this._background, e.cachedCanvases);
    return s !== vn.SHADING && (t.setTransform(...e.baseTransform), this.matrix && t.transform(...this.matrix)), t.translate(l.offsetX, l.offsetY), t.scale(l.scaleX, l.scaleY), t.createPattern(l.canvas, "no-repeat");
  }
}
class $k extends mA {
  getPattern() {
    return "hotpink";
  }
}
function Vk(p) {
  switch (p[0]) {
    case "RadialAxial":
      return new Uk(p);
    case "Mesh":
      return new jk(p);
    case "Dummy":
      return new $k();
  }
  throw new Error(`Unknown IR type: ${p[0]}`);
}
const $E = {
  COLORED: 1,
  UNCOLORED: 2
}, cy = class cy {
  constructor(t, e, n, s) {
    this.color = t[1], this.operatorList = t[2], this.matrix = t[3], this.bbox = t[4], this.xstep = t[5], this.ystep = t[6], this.paintType = t[7], this.tilingType = t[8], this.ctx = e, this.canvasGraphicsFactory = n, this.baseTransform = s;
  }
  createPatternCanvas(t, e) {
    var ot, st;
    const {
      bbox: n,
      operatorList: s,
      paintType: a,
      tilingType: l,
      color: u,
      canvasGraphicsFactory: h
    } = this;
    let {
      xstep: f,
      ystep: g
    } = this;
    f = Math.abs(f), g = Math.abs(g), Sy("TilingType: " + l);
    const y = n[0], v = n[1], S = n[2], x = n[3], _ = S - y, k = x - v, T = new Float32Array(2);
    ft.singularValueDecompose2dScale(this.matrix, T);
    const [R, P] = T;
    ft.singularValueDecompose2dScale(this.baseTransform, T);
    const L = R * T[0], D = P * T[1];
    let N = _, B = k, $ = !1, V = !1;
    const X = Math.ceil(f * L), q = Math.ceil(g * D), it = Math.ceil(_ * L), lt = Math.ceil(k * D);
    X >= it ? N = f : $ = !0, q >= lt ? B = g : V = !0;
    const rt = this.getSizeAndScale(N, this.ctx.canvas.width, L), vt = this.getSizeAndScale(B, this.ctx.canvas.height, D), tt = t.cachedCanvases.getCanvas("pattern", rt.size, vt.size), Q = tt.context, W = h.createCanvasGraphics(Q, e);
    if (W.groupLevel = t.groupLevel, this.setFillAndStrokeStyleToContext(W, a, u), Q.translate(-rt.scale * y, -vt.scale * v), W.transform(0, rt.scale, 0, 0, vt.scale, 0, 0), Q.save(), (ot = W.dependencyTracker) == null || ot.save(), this.clipBbox(W, y, v, S, x), W.baseTransform = _e(W.ctx), W.executeOperatorList(s), W.endDrawing(), (st = W.dependencyTracker) == null || st.restore(), Q.restore(), $ || V) {
      const I = tt.canvas;
      $ && (N = f), V && (B = g);
      const z = this.getSizeAndScale(N, this.ctx.canvas.width, L), K = this.getSizeAndScale(B, this.ctx.canvas.height, D), ct = z.size, Et = K.size, xt = t.cachedCanvases.getCanvas("pattern-workaround", ct, Et), bt = xt.context, Ot = $ ? Math.floor(_ / f) : 0, Bt = V ? Math.floor(k / g) : 0;
      for (let Dt = 0; Dt <= Ot; Dt++)
        for (let Qe = 0; Qe <= Bt; Qe++)
          bt.drawImage(I, ct * Dt, Et * Qe, ct, Et, 0, 0, ct, Et);
      return {
        canvas: xt.canvas,
        scaleX: z.scale,
        scaleY: K.scale,
        offsetX: y,
        offsetY: v
      };
    }
    return {
      canvas: tt.canvas,
      scaleX: rt.scale,
      scaleY: vt.scale,
      offsetX: y,
      offsetY: v
    };
  }
  getSizeAndScale(t, e, n) {
    const s = Math.max(cy.MAX_PATTERN_SIZE, e);
    let a = Math.ceil(t * n);
    return a >= s ? a = s : n = a / t, {
      scale: n,
      size: a
    };
  }
  clipBbox(t, e, n, s, a) {
    const l = s - e, u = a - n;
    t.ctx.rect(e, n, l, u), ft.axialAlignedBoundingBox([e, n, s, a], _e(t.ctx), t.current.minMax), t.clip(), t.endPath();
  }
  setFillAndStrokeStyleToContext(t, e, n) {
    const s = t.ctx, a = t.current;
    switch (e) {
      case $E.COLORED:
        const {
          fillStyle: l,
          strokeStyle: u
        } = this.ctx;
        s.fillStyle = a.fillColor = l, s.strokeStyle = a.strokeColor = u;
        break;
      case $E.UNCOLORED:
        s.fillStyle = s.strokeStyle = n, a.fillColor = a.strokeColor = n;
        break;
      default:
        throw new hk(`Unsupported paint type: ${e}`);
    }
  }
  isModifyingCurrentTransform() {
    return !1;
  }
  getPattern(t, e, n, s, a) {
    let l = n;
    s !== vn.SHADING && (l = ft.transform(l, e.baseTransform), this.matrix && (l = ft.transform(l, this.matrix)));
    const u = this.createPatternCanvas(e, a);
    let h = new DOMMatrix(l);
    h = h.translate(u.offsetX, u.offsetY), h = h.scale(1 / u.scaleX, 1 / u.scaleY);
    const f = t.createPattern(u.canvas, "repeat");
    return f.setTransform(h), f;
  }
};
Z(cy, "MAX_PATTERN_SIZE", 3e3);
let cw = cy;
function Wk({
  src: p,
  srcPos: t = 0,
  dest: e,
  width: n,
  height: s,
  nonBlackColor: a = 4294967295,
  inverseDecode: l = !1
}) {
  const u = cn.isLittleEndian ? 4278190080 : 255, [h, f] = l ? [a, u] : [u, a], g = n >> 3, y = n & 7, v = p.length;
  e = new Uint32Array(e.buffer);
  let S = 0;
  for (let x = 0; x < s; x++) {
    for (const k = t + g; t < k; t++) {
      const T = t < v ? p[t] : 255;
      e[S++] = T & 128 ? f : h, e[S++] = T & 64 ? f : h, e[S++] = T & 32 ? f : h, e[S++] = T & 16 ? f : h, e[S++] = T & 8 ? f : h, e[S++] = T & 4 ? f : h, e[S++] = T & 2 ? f : h, e[S++] = T & 1 ? f : h;
    }
    if (y === 0)
      continue;
    const _ = t < v ? p[t++] : 255;
    for (let k = 0; k < y; k++)
      e[S++] = _ & 1 << 7 - k ? f : h;
  }
  return {
    srcPos: t,
    destPos: S
  };
}
const VE = 16, WE = 100, Gk = 15, GE = 10, ri = 16, P0 = new DOMMatrix(), Pi = new Float32Array(2), Yc = new Float32Array([1 / 0, 1 / 0, -1 / 0, -1 / 0]);
function Xk(p, t) {
  if (p._removeMirroring)
    throw new Error("Context is already forwarding operations.");
  p.__originalSave = p.save, p.__originalRestore = p.restore, p.__originalRotate = p.rotate, p.__originalScale = p.scale, p.__originalTranslate = p.translate, p.__originalTransform = p.transform, p.__originalSetTransform = p.setTransform, p.__originalResetTransform = p.resetTransform, p.__originalClip = p.clip, p.__originalMoveTo = p.moveTo, p.__originalLineTo = p.lineTo, p.__originalBezierCurveTo = p.bezierCurveTo, p.__originalRect = p.rect, p.__originalClosePath = p.closePath, p.__originalBeginPath = p.beginPath, p._removeMirroring = () => {
    p.save = p.__originalSave, p.restore = p.__originalRestore, p.rotate = p.__originalRotate, p.scale = p.__originalScale, p.translate = p.__originalTranslate, p.transform = p.__originalTransform, p.setTransform = p.__originalSetTransform, p.resetTransform = p.__originalResetTransform, p.clip = p.__originalClip, p.moveTo = p.__originalMoveTo, p.lineTo = p.__originalLineTo, p.bezierCurveTo = p.__originalBezierCurveTo, p.rect = p.__originalRect, p.closePath = p.__originalClosePath, p.beginPath = p.__originalBeginPath, delete p._removeMirroring;
  }, p.save = function() {
    t.save(), this.__originalSave();
  }, p.restore = function() {
    t.restore(), this.__originalRestore();
  }, p.translate = function(e, n) {
    t.translate(e, n), this.__originalTranslate(e, n);
  }, p.scale = function(e, n) {
    t.scale(e, n), this.__originalScale(e, n);
  }, p.transform = function(e, n, s, a, l, u) {
    t.transform(e, n, s, a, l, u), this.__originalTransform(e, n, s, a, l, u);
  }, p.setTransform = function(e, n, s, a, l, u) {
    t.setTransform(e, n, s, a, l, u), this.__originalSetTransform(e, n, s, a, l, u);
  }, p.resetTransform = function() {
    t.resetTransform(), this.__originalResetTransform();
  }, p.rotate = function(e) {
    t.rotate(e), this.__originalRotate(e);
  }, p.clip = function(e) {
    t.clip(e), this.__originalClip(e);
  }, p.moveTo = function(e, n) {
    t.moveTo(e, n), this.__originalMoveTo(e, n);
  }, p.lineTo = function(e, n) {
    t.lineTo(e, n), this.__originalLineTo(e, n);
  }, p.bezierCurveTo = function(e, n, s, a, l, u) {
    t.bezierCurveTo(e, n, s, a, l, u), this.__originalBezierCurveTo(e, n, s, a, l, u);
  }, p.rect = function(e, n, s, a) {
    t.rect(e, n, s, a), this.__originalRect(e, n, s, a);
  }, p.closePath = function() {
    t.closePath(), this.__originalClosePath();
  }, p.beginPath = function() {
    t.beginPath(), this.__originalBeginPath();
  };
}
class qk {
  constructor(t) {
    this.canvasFactory = t, this.cache = /* @__PURE__ */ Object.create(null);
  }
  getCanvas(t, e, n) {
    let s;
    return this.cache[t] !== void 0 ? (s = this.cache[t], this.canvasFactory.reset(s, e, n)) : (s = this.canvasFactory.create(e, n), this.cache[t] = s), s;
  }
  delete(t) {
    delete this.cache[t];
  }
  clear() {
    for (const t in this.cache) {
      const e = this.cache[t];
      this.canvasFactory.destroy(e), delete this.cache[t];
    }
  }
}
function om(p, t, e, n, s, a, l, u, h, f) {
  const [g, y, v, S, x, _] = _e(p);
  if (y === 0 && v === 0) {
    const R = l * g + x, P = Math.round(R), L = u * S + _, D = Math.round(L), N = (l + h) * g + x, B = Math.abs(Math.round(N) - P) || 1, $ = (u + f) * S + _, V = Math.abs(Math.round($) - D) || 1;
    return p.setTransform(Math.sign(g), 0, 0, Math.sign(S), P, D), p.drawImage(t, e, n, s, a, 0, 0, B, V), p.setTransform(g, y, v, S, x, _), [B, V];
  }
  if (g === 0 && S === 0) {
    const R = u * v + x, P = Math.round(R), L = l * y + _, D = Math.round(L), N = (u + f) * v + x, B = Math.abs(Math.round(N) - P) || 1, $ = (l + h) * y + _, V = Math.abs(Math.round($) - D) || 1;
    return p.setTransform(0, Math.sign(y), Math.sign(v), 0, P, D), p.drawImage(t, e, n, s, a, 0, 0, V, B), p.setTransform(g, y, v, S, x, _), [V, B];
  }
  p.drawImage(t, e, n, s, a, l, u, h, f);
  const k = Math.hypot(g, y), T = Math.hypot(v, S);
  return [k * h, T * f];
}
class XE {
  constructor(t, e, n) {
    Z(this, "alphaIsShape", !1);
    Z(this, "fontSize", 0);
    Z(this, "fontSizeScale", 1);
    Z(this, "textMatrix", null);
    Z(this, "textMatrixScale", 1);
    Z(this, "fontMatrix", F0);
    Z(this, "leading", 0);
    Z(this, "x", 0);
    Z(this, "y", 0);
    Z(this, "lineX", 0);
    Z(this, "lineY", 0);
    Z(this, "charSpacing", 0);
    Z(this, "wordSpacing", 0);
    Z(this, "textHScale", 1);
    Z(this, "textRenderingMode", dn.FILL);
    Z(this, "textRise", 0);
    Z(this, "fillColor", "#000000");
    Z(this, "strokeColor", "#000000");
    Z(this, "patternFill", !1);
    Z(this, "patternStroke", !1);
    Z(this, "fillAlpha", 1);
    Z(this, "strokeAlpha", 1);
    Z(this, "lineWidth", 1);
    Z(this, "activeSMask", null);
    Z(this, "transferMaps", "none");
    n == null || n(this), this.clipBox = new Float32Array([0, 0, t, e]), this.minMax = Yc.slice();
  }
  clone() {
    const t = Object.create(this);
    return t.clipBox = this.clipBox.slice(), t.minMax = this.minMax.slice(), t;
  }
  getPathBoundingBox(t = vn.FILL, e = null) {
    const n = this.minMax.slice();
    if (t === vn.STROKE) {
      e || ce("Stroke bounding box must include transform."), ft.singularValueDecompose2dScale(e, Pi);
      const s = Pi[0] * this.lineWidth / 2, a = Pi[1] * this.lineWidth / 2;
      n[0] -= s, n[1] -= a, n[2] += s, n[3] += a;
    }
    return n;
  }
  updateClipFromPath() {
    const t = ft.intersect(this.clipBox, this.getPathBoundingBox());
    this.startNewPathAndClipBox(t || [0, 0, 0, 0]);
  }
  isEmptyClip() {
    return this.minMax[0] === 1 / 0;
  }
  startNewPathAndClipBox(t) {
    this.clipBox.set(t, 0), this.minMax.set(Yc, 0);
  }
  getClippedPathBoundingBox(t = vn.FILL, e = null) {
    return ft.intersect(this.clipBox, this.getPathBoundingBox(t, e));
  }
}
function qE(p, t) {
  if (t instanceof ImageData) {
    p.putImageData(t, 0, 0);
    return;
  }
  const e = t.height, n = t.width, s = e % ri, a = (e - s) / ri, l = s === 0 ? a : a + 1, u = p.createImageData(n, ri);
  let h = 0, f;
  const g = t.data, y = u.data;
  let v, S, x, _;
  if (t.kind === Vd.GRAYSCALE_1BPP) {
    const k = g.byteLength, T = new Uint32Array(y.buffer, 0, y.byteLength >> 2), R = T.length, P = n + 7 >> 3, L = 4294967295, D = cn.isLittleEndian ? 4278190080 : 255;
    for (v = 0; v < l; v++) {
      for (x = v < a ? ri : s, f = 0, S = 0; S < x; S++) {
        const N = k - h;
        let B = 0;
        const $ = N > P ? n : N * 8 - 7, V = $ & -8;
        let X = 0, q = 0;
        for (; B < V; B += 8)
          q = g[h++], T[f++] = q & 128 ? L : D, T[f++] = q & 64 ? L : D, T[f++] = q & 32 ? L : D, T[f++] = q & 16 ? L : D, T[f++] = q & 8 ? L : D, T[f++] = q & 4 ? L : D, T[f++] = q & 2 ? L : D, T[f++] = q & 1 ? L : D;
        for (; B < $; B++)
          X === 0 && (q = g[h++], X = 128), T[f++] = q & X ? L : D, X >>= 1;
      }
      for (; f < R; )
        T[f++] = 0;
      p.putImageData(u, 0, v * ri);
    }
  } else if (t.kind === Vd.RGBA_32BPP) {
    for (S = 0, _ = n * ri * 4, v = 0; v < a; v++)
      y.set(g.subarray(h, h + _)), h += _, p.putImageData(u, 0, S), S += ri;
    v < l && (_ = n * s * 4, y.set(g.subarray(h, h + _)), p.putImageData(u, 0, S));
  } else if (t.kind === Vd.RGB_24BPP)
    for (x = ri, _ = n * x, v = 0; v < l; v++) {
      for (v >= a && (x = s, _ = n * x), f = 0, S = _; S--; )
        y[f++] = g[h++], y[f++] = g[h++], y[f++] = g[h++], y[f++] = 255;
      p.putImageData(u, 0, v * ri);
    }
  else
    throw new Error(`bad image kind: ${t.kind}`);
}
function YE(p, t) {
  if (t.bitmap) {
    p.drawImage(t.bitmap, 0, 0);
    return;
  }
  const e = t.height, n = t.width, s = e % ri, a = (e - s) / ri, l = s === 0 ? a : a + 1, u = p.createImageData(n, ri);
  let h = 0;
  const f = t.data, g = u.data;
  for (let y = 0; y < l; y++) {
    const v = y < a ? ri : s;
    ({
      srcPos: h
    } = Wk({
      src: f,
      srcPos: h,
      dest: g,
      width: n,
      height: v,
      nonBlackColor: 0
    })), p.putImageData(u, 0, y * ri);
  }
}
function wd(p, t) {
  const e = ["strokeStyle", "fillStyle", "fillRule", "globalAlpha", "lineWidth", "lineCap", "lineJoin", "miterLimit", "globalCompositeOperation", "font", "filter"];
  for (const n of e)
    p[n] !== void 0 && (t[n] = p[n]);
  p.setLineDash !== void 0 && (t.setLineDash(p.getLineDash()), t.lineDashOffset = p.lineDashOffset);
}
function am(p) {
  p.strokeStyle = p.fillStyle = "#000000", p.fillRule = "nonzero", p.globalAlpha = 1, p.lineWidth = 1, p.lineCap = "butt", p.lineJoin = "miter", p.miterLimit = 10, p.globalCompositeOperation = "source-over", p.font = "10px sans-serif", p.setLineDash !== void 0 && (p.setLineDash([]), p.lineDashOffset = 0);
  const {
    filter: t
  } = p;
  t !== "none" && t !== "" && (p.filter = "none");
}
function KE(p, t) {
  if (t)
    return !0;
  ft.singularValueDecompose2dScale(p, Pi);
  const e = Math.fround(ks.pixelRatio * lo.PDF_TO_CSS_UNITS);
  return Pi[0] <= e && Pi[1] <= e;
}
const Yk = ["butt", "round", "square"], Kk = ["miter", "round", "bevel"], Qk = {}, QE = {};
var Ps, uw, hw, dw;
const TA = class TA {
  constructor(t, e, n, s, a, {
    optionalContentConfig: l,
    markedContentStack: u = null
  }, h, f, g) {
    b(this, Ps);
    this.ctx = t, this.current = new XE(this.ctx.canvas.width, this.ctx.canvas.height), this.stateStack = [], this.pendingClip = null, this.pendingEOFill = !1, this.res = null, this.xobjs = null, this.commonObjs = e, this.objs = n, this.canvasFactory = s, this.filterFactory = a, this.groupStack = [], this.baseTransform = null, this.baseTransformStack = [], this.groupLevel = 0, this.smaskStack = [], this.smaskCounter = 0, this.tempSMask = null, this.suspendedCtx = null, this.contentVisible = !0, this.markedContentStack = u || [], this.optionalContentConfig = l, this.cachedCanvases = new qk(this.canvasFactory), this.cachedPatterns = /* @__PURE__ */ new Map(), this.annotationCanvasMap = h, this.viewportScale = 1, this.outputScaleX = 1, this.outputScaleY = 1, this.pageColors = f, this._cachedScaleForStroking = [-1, 0], this._cachedGetSinglePixelWidth = null, this._cachedBitmapsMap = /* @__PURE__ */ new Map(), this.dependencyTracker = g ?? null;
  }
  getObject(t, e, n = null) {
    var s;
    return typeof e == "string" ? ((s = this.dependencyTracker) == null || s.recordNamedDependency(t, e), e.startsWith("g_") ? this.commonObjs.get(e) : this.objs.get(e)) : n;
  }
  beginDrawing({
    transform: t,
    viewport: e,
    transparency: n = !1,
    background: s = null
  }) {
    const a = this.ctx.canvas.width, l = this.ctx.canvas.height, u = this.ctx.fillStyle;
    if (this.ctx.fillStyle = s || "#ffffff", this.ctx.fillRect(0, 0, a, l), this.ctx.fillStyle = u, n) {
      const h = this.cachedCanvases.getCanvas("transparent", a, l);
      this.compositeCtx = this.ctx, this.transparentCanvas = h.canvas, this.ctx = h.context, this.ctx.save(), this.ctx.transform(..._e(this.compositeCtx));
    }
    this.ctx.save(), am(this.ctx), t && (this.ctx.transform(...t), this.outputScaleX = t[0], this.outputScaleY = t[0]), this.ctx.transform(...e.transform), this.viewportScale = e.scale, this.baseTransform = _e(this.ctx);
  }
  executeOperatorList(t, e, n, s, a) {
    var T;
    const l = t.argsArray, u = t.fnArray;
    let h = e || 0;
    const f = l.length;
    if (f === h)
      return h;
    const g = f - h > GE && typeof n == "function", y = g ? Date.now() + Gk : 0;
    let v = 0;
    const S = this.commonObjs, x = this.objs;
    let _, k;
    for (; ; ) {
      if (s !== void 0 && h === s.nextBreakPoint)
        return s.breakIt(h, n), h;
      if (!a || a(h))
        if (_ = u[h], k = l[h] ?? null, _ !== Ph.dependency)
          k === null ? this[_](h) : this[_](h, ...k);
        else
          for (const R of k) {
            (T = this.dependencyTracker) == null || T.recordNamedData(R, h);
            const P = R.startsWith("g_") ? S : x;
            if (!P.has(R))
              return P.get(R, n), h;
          }
      if (h++, h === f)
        return h;
      if (g && ++v > GE) {
        if (Date.now() > y)
          return n(), h;
        v = 0;
      }
    }
  }
  endDrawing() {
    E(this, Ps, uw).call(this), this.cachedCanvases.clear(), this.cachedPatterns.clear();
    for (const t of this._cachedBitmapsMap.values()) {
      for (const e of t.values())
        typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement && (e.width = e.height = 0);
      t.clear();
    }
    this._cachedBitmapsMap.clear(), E(this, Ps, hw).call(this);
  }
  _scaleImage(t, e) {
    const n = t.width ?? t.displayWidth, s = t.height ?? t.displayHeight;
    let a = Math.max(Math.hypot(e[0], e[1]), 1), l = Math.max(Math.hypot(e[2], e[3]), 1), u = n, h = s, f = "prescale1", g, y;
    for (; a > 2 && u > 1 || l > 2 && h > 1; ) {
      let v = u, S = h;
      a > 2 && u > 1 && (v = u >= 16384 ? Math.floor(u / 2) - 1 || 1 : Math.ceil(u / 2), a /= u / v), l > 2 && h > 1 && (S = h >= 16384 ? Math.floor(h / 2) - 1 || 1 : Math.ceil(h) / 2, l /= h / S), g = this.cachedCanvases.getCanvas(f, v, S), y = g.context, y.clearRect(0, 0, v, S), y.drawImage(t, 0, 0, u, h, 0, 0, v, S), t = g.canvas, u = v, h = S, f = f === "prescale1" ? "prescale2" : "prescale1";
    }
    return {
      img: t,
      paintWidth: u,
      paintHeight: h
    };
  }
  _createMaskCanvas(t, e) {
    var X, q;
    const n = this.ctx, {
      width: s,
      height: a
    } = e, l = this.current.fillColor, u = this.current.patternFill, h = _e(n);
    let f, g, y, v;
    if ((e.bitmap || e.data) && e.count > 1) {
      const it = e.bitmap || e.data.buffer;
      g = JSON.stringify(u ? h : [h.slice(0, 4), l]), f = this._cachedBitmapsMap.get(it), f || (f = /* @__PURE__ */ new Map(), this._cachedBitmapsMap.set(it, f));
      const lt = f.get(g);
      if (lt && !u) {
        const rt = Math.round(Math.min(h[0], h[2]) + h[4]), vt = Math.round(Math.min(h[1], h[3]) + h[5]);
        return (X = this.dependencyTracker) == null || X.recordDependencies(t, ji.transformAndFill), {
          canvas: lt,
          offsetX: rt,
          offsetY: vt
        };
      }
      y = lt;
    }
    y || (v = this.cachedCanvases.getCanvas("maskCanvas", s, a), YE(v.context, e));
    let S = ft.transform(h, [1 / s, 0, 0, -1 / a, 0, 0]);
    S = ft.transform(S, [1, 0, 0, 1, 0, -a]);
    const x = Yc.slice();
    ft.axialAlignedBoundingBox([0, 0, s, a], S, x);
    const [_, k, T, R] = x, P = Math.round(T - _) || 1, L = Math.round(R - k) || 1, D = this.cachedCanvases.getCanvas("fillCanvas", P, L), N = D.context, B = _, $ = k;
    N.translate(-B, -$), N.transform(...S), y || (y = this._scaleImage(v.canvas, Ns(N)), y = y.img, f && u && f.set(g, y)), N.imageSmoothingEnabled = KE(_e(N), e.interpolate), om(N, y, 0, 0, y.width, y.height, 0, 0, s, a), N.globalCompositeOperation = "source-in";
    const V = ft.transform(Ns(N), [1, 0, 0, 1, -B, -$]);
    return N.fillStyle = u ? l.getPattern(n, this, V, vn.FILL, t) : l, N.fillRect(0, 0, s, a), f && !u && (this.cachedCanvases.delete("fillCanvas"), f.set(g, D.canvas)), (q = this.dependencyTracker) == null || q.recordDependencies(t, ji.transformAndFill), {
      canvas: D.canvas,
      offsetX: Math.round(B),
      offsetY: Math.round($)
    };
  }
  setLineWidth(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("lineWidth", t), e !== this.current.lineWidth && (this._cachedScaleForStroking[0] = -1), this.current.lineWidth = e, this.ctx.lineWidth = e;
  }
  setLineCap(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("lineCap", t), this.ctx.lineCap = Yk[e];
  }
  setLineJoin(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("lineJoin", t), this.ctx.lineJoin = Kk[e];
  }
  setMiterLimit(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("miterLimit", t), this.ctx.miterLimit = e;
  }
  setDash(t, e, n) {
    var a;
    (a = this.dependencyTracker) == null || a.recordSimpleData("dash", t);
    const s = this.ctx;
    s.setLineDash !== void 0 && (s.setLineDash(e), s.lineDashOffset = n);
  }
  setRenderingIntent(t, e) {
  }
  setFlatness(t, e) {
  }
  setGState(t, e) {
    var n, s, a, l, u;
    for (const [h, f] of e)
      switch (h) {
        case "LW":
          this.setLineWidth(t, f);
          break;
        case "LC":
          this.setLineCap(t, f);
          break;
        case "LJ":
          this.setLineJoin(t, f);
          break;
        case "ML":
          this.setMiterLimit(t, f);
          break;
        case "D":
          this.setDash(t, f[0], f[1]);
          break;
        case "RI":
          this.setRenderingIntent(t, f);
          break;
        case "FL":
          this.setFlatness(t, f);
          break;
        case "Font":
          this.setFont(t, f[0], f[1]);
          break;
        case "CA":
          (n = this.dependencyTracker) == null || n.recordSimpleData("strokeAlpha", t), this.current.strokeAlpha = f;
          break;
        case "ca":
          (s = this.dependencyTracker) == null || s.recordSimpleData("fillAlpha", t), this.ctx.globalAlpha = this.current.fillAlpha = f;
          break;
        case "BM":
          (a = this.dependencyTracker) == null || a.recordSimpleData("globalCompositeOperation", t), this.ctx.globalCompositeOperation = f;
          break;
        case "SMask":
          (l = this.dependencyTracker) == null || l.recordSimpleData("SMask", t), this.current.activeSMask = f ? this.tempSMask : null, this.tempSMask = null, this.checkSMaskState();
          break;
        case "TR":
          (u = this.dependencyTracker) == null || u.recordSimpleData("filter", t), this.ctx.filter = this.current.transferMaps = this.filterFactory.addFilter(f);
          break;
      }
  }
  get inSMaskMode() {
    return !!this.suspendedCtx;
  }
  checkSMaskState() {
    const t = this.inSMaskMode;
    this.current.activeSMask && !t ? this.beginSMaskMode() : !this.current.activeSMask && t && this.endSMaskMode();
  }
  beginSMaskMode(t) {
    if (this.inSMaskMode)
      throw new Error("beginSMaskMode called while already in smask mode");
    const e = this.ctx.canvas.width, n = this.ctx.canvas.height, s = "smaskGroupAt" + this.groupLevel, a = this.cachedCanvases.getCanvas(s, e, n);
    this.suspendedCtx = this.ctx;
    const l = this.ctx = a.context;
    l.setTransform(this.suspendedCtx.getTransform()), wd(this.suspendedCtx, l), Xk(l, this.suspendedCtx), this.setGState(t, [["BM", "source-over"]]);
  }
  endSMaskMode() {
    if (!this.inSMaskMode)
      throw new Error("endSMaskMode called while not in smask mode");
    this.ctx._removeMirroring(), wd(this.ctx, this.suspendedCtx), this.ctx = this.suspendedCtx, this.suspendedCtx = null;
  }
  compose(t) {
    if (!this.current.activeSMask)
      return;
    t ? (t[0] = Math.floor(t[0]), t[1] = Math.floor(t[1]), t[2] = Math.ceil(t[2]), t[3] = Math.ceil(t[3])) : t = [0, 0, this.ctx.canvas.width, this.ctx.canvas.height];
    const e = this.current.activeSMask, n = this.suspendedCtx;
    this.composeSMask(n, e, this.ctx, t), this.ctx.save(), this.ctx.setTransform(1, 0, 0, 1, 0, 0), this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height), this.ctx.restore();
  }
  composeSMask(t, e, n, s) {
    const a = s[0], l = s[1], u = s[2] - a, h = s[3] - l;
    u === 0 || h === 0 || (this.genericComposeSMask(e.context, n, u, h, e.subtype, e.backdrop, e.transferMap, a, l, e.offsetX, e.offsetY), t.save(), t.globalAlpha = 1, t.globalCompositeOperation = "source-over", t.setTransform(1, 0, 0, 1, 0, 0), t.drawImage(n.canvas, 0, 0), t.restore());
  }
  genericComposeSMask(t, e, n, s, a, l, u, h, f, g, y) {
    let v = t.canvas, S = h - g, x = f - y;
    if (l)
      if (S < 0 || x < 0 || S + n > v.width || x + s > v.height) {
        const k = this.cachedCanvases.getCanvas("maskExtension", n, s), T = k.context;
        T.drawImage(v, -S, -x), T.globalCompositeOperation = "destination-atop", T.fillStyle = l, T.fillRect(0, 0, n, s), T.globalCompositeOperation = "source-over", v = k.canvas, S = x = 0;
      } else {
        t.save(), t.globalAlpha = 1, t.setTransform(1, 0, 0, 1, 0, 0);
        const k = new Path2D();
        k.rect(S, x, n, s), t.clip(k), t.globalCompositeOperation = "destination-atop", t.fillStyle = l, t.fillRect(S, x, n, s), t.restore();
      }
    e.save(), e.globalAlpha = 1, e.setTransform(1, 0, 0, 1, 0, 0), a === "Alpha" && u ? e.filter = this.filterFactory.addAlphaFilter(u) : a === "Luminosity" && (e.filter = this.filterFactory.addLuminosityFilter(u));
    const _ = new Path2D();
    _.rect(h, f, n, s), e.clip(_), e.globalCompositeOperation = "destination-in", e.drawImage(v, S, x, n, s, h, f, n, s), e.restore();
  }
  save(t) {
    var n;
    this.inSMaskMode && wd(this.ctx, this.suspendedCtx), this.ctx.save();
    const e = this.current;
    this.stateStack.push(e), this.current = e.clone(), (n = this.dependencyTracker) == null || n.save(t);
  }
  restore(t) {
    var e;
    if ((e = this.dependencyTracker) == null || e.restore(t), this.stateStack.length === 0) {
      this.inSMaskMode && this.endSMaskMode();
      return;
    }
    this.current = this.stateStack.pop(), this.ctx.restore(), this.inSMaskMode && wd(this.suspendedCtx, this.ctx), this.checkSMaskState(), this.pendingClip = null, this._cachedScaleForStroking[0] = -1, this._cachedGetSinglePixelWidth = null;
  }
  transform(t, e, n, s, a, l, u) {
    var h;
    (h = this.dependencyTracker) == null || h.recordIncrementalData("transform", t), this.ctx.transform(e, n, s, a, l, u), this._cachedScaleForStroking[0] = -1, this._cachedGetSinglePixelWidth = null;
  }
  constructPath(t, e, n, s) {
    let [a] = n;
    if (!s) {
      a || (a = n[0] = new Path2D()), this[e](t, a);
      return;
    }
    if (this.dependencyTracker !== null) {
      const l = e === Ph.stroke ? this.current.lineWidth / 2 : 0;
      this.dependencyTracker.resetBBox(t).recordBBox(t, this.ctx, s[0] - l, s[2] + l, s[1] - l, s[3] + l).recordDependencies(t, ["transform"]);
    }
    if (!(a instanceof Path2D)) {
      const l = n[0] = new Path2D();
      for (let u = 0, h = a.length; u < h; )
        switch (a[u++]) {
          case tm.moveTo:
            l.moveTo(a[u++], a[u++]);
            break;
          case tm.lineTo:
            l.lineTo(a[u++], a[u++]);
            break;
          case tm.curveTo:
            l.bezierCurveTo(a[u++], a[u++], a[u++], a[u++], a[u++], a[u++]);
            break;
          case tm.closePath:
            l.closePath();
            break;
          default:
            Rt(`Unrecognized drawing path operator: ${a[u - 1]}`);
            break;
        }
      a = l;
    }
    ft.axialAlignedBoundingBox(s, _e(this.ctx), this.current.minMax), this[e](t, a), this._pathStartIdx = t;
  }
  closePath(t) {
    this.ctx.closePath();
  }
  stroke(t, e, n = !0) {
    var l;
    const s = this.ctx, a = this.current.strokeColor;
    if (s.globalAlpha = this.current.strokeAlpha, this.contentVisible)
      if (typeof a == "object" && (a != null && a.getPattern)) {
        const u = a.isModifyingCurrentTransform() ? s.getTransform() : null;
        if (s.save(), s.strokeStyle = a.getPattern(s, this, Ns(s), vn.STROKE, t), u) {
          const h = new Path2D();
          h.addPath(e, s.getTransform().invertSelf().multiplySelf(u)), e = h;
        }
        this.rescaleAndStroke(e, !1), s.restore();
      } else
        this.rescaleAndStroke(e, !0);
    (l = this.dependencyTracker) == null || l.recordDependencies(t, ji.stroke), n && this.consumePath(t, e, this.current.getClippedPathBoundingBox(vn.STROKE, _e(this.ctx))), s.globalAlpha = this.current.fillAlpha;
  }
  closeStroke(t, e) {
    this.stroke(t, e);
  }
  fill(t, e, n = !0) {
    var f, g, y;
    const s = this.ctx, a = this.current.fillColor, l = this.current.patternFill;
    let u = !1;
    if (l) {
      const v = a.isModifyingCurrentTransform() ? s.getTransform() : null;
      if ((f = this.dependencyTracker) == null || f.save(t), s.save(), s.fillStyle = a.getPattern(s, this, Ns(s), vn.FILL, t), v) {
        const S = new Path2D();
        S.addPath(e, s.getTransform().invertSelf().multiplySelf(v)), e = S;
      }
      u = !0;
    }
    const h = this.current.getClippedPathBoundingBox();
    this.contentVisible && h !== null && (this.pendingEOFill ? (s.fill(e, "evenodd"), this.pendingEOFill = !1) : s.fill(e)), (g = this.dependencyTracker) == null || g.recordDependencies(t, ji.fill), u && (s.restore(), (y = this.dependencyTracker) == null || y.restore(t)), n && this.consumePath(t, e, h);
  }
  eoFill(t, e) {
    this.pendingEOFill = !0, this.fill(t, e);
  }
  fillStroke(t, e) {
    this.fill(t, e, !1), this.stroke(t, e, !1), this.consumePath(t, e);
  }
  eoFillStroke(t, e) {
    this.pendingEOFill = !0, this.fillStroke(t, e);
  }
  closeFillStroke(t, e) {
    this.fillStroke(t, e);
  }
  closeEOFillStroke(t, e) {
    this.pendingEOFill = !0, this.fillStroke(t, e);
  }
  endPath(t, e) {
    this.consumePath(t, e);
  }
  rawFillPath(t, e) {
    var n;
    this.ctx.fill(e), (n = this.dependencyTracker) == null || n.recordDependencies(t, ji.rawFillPath).recordOperation(t);
  }
  clip(t) {
    var e;
    (e = this.dependencyTracker) == null || e.recordFutureForcedDependency("clipMode", t), this.pendingClip = Qk;
  }
  eoClip(t) {
    var e;
    (e = this.dependencyTracker) == null || e.recordFutureForcedDependency("clipMode", t), this.pendingClip = QE;
  }
  beginText(t) {
    var e;
    this.current.textMatrix = null, this.current.textMatrixScale = 1, this.current.x = this.current.lineX = 0, this.current.y = this.current.lineY = 0, (e = this.dependencyTracker) == null || e.recordOpenMarker(t).resetIncrementalData("sameLineText").resetIncrementalData("moveText", t);
  }
  endText(t) {
    const e = this.pendingTextPaths, n = this.ctx;
    if (this.dependencyTracker) {
      const {
        dependencyTracker: s
      } = this;
      e !== void 0 && s.recordFutureForcedDependency("textClip", s.getOpenMarker()).recordFutureForcedDependency("textClip", t), s.recordCloseMarker(t);
    }
    if (e !== void 0) {
      const s = new Path2D(), a = n.getTransform().invertSelf();
      for (const {
        transform: l,
        x: u,
        y: h,
        fontSize: f,
        path: g
      } of e)
        g && s.addPath(g, new DOMMatrix(l).preMultiplySelf(a).translate(u, h).scale(f, -f));
      n.clip(s);
    }
    delete this.pendingTextPaths;
  }
  setCharSpacing(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("charSpacing", t), this.current.charSpacing = e;
  }
  setWordSpacing(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("wordSpacing", t), this.current.wordSpacing = e;
  }
  setHScale(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("hScale", t), this.current.textHScale = e / 100;
  }
  setLeading(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("leading", t), this.current.leading = -e;
  }
  setFont(t, e, n) {
    var y, v;
    (y = this.dependencyTracker) == null || y.recordSimpleData("font", t).recordSimpleDataFromNamed("fontObj", e, t);
    const s = this.commonObjs.get(e), a = this.current;
    if (!s)
      throw new Error(`Can't find font for ${e}`);
    if (a.fontMatrix = s.fontMatrix || F0, (a.fontMatrix[0] === 0 || a.fontMatrix[3] === 0) && Rt("Invalid font matrix for font " + e), n < 0 ? (n = -n, a.fontDirection = -1) : a.fontDirection = 1, this.current.font = s, this.current.fontSize = n, s.isType3Font)
      return;
    const l = s.loadedName || "sans-serif", u = ((v = s.systemFontInfo) == null ? void 0 : v.css) || `"${l}", ${s.fallbackName}`;
    let h = "normal";
    s.black ? h = "900" : s.bold && (h = "bold");
    const f = s.italic ? "italic" : "normal";
    let g = n;
    n < VE ? g = VE : n > WE && (g = WE), this.current.fontSizeScale = n / g, this.ctx.font = `${f} ${h} ${g}px ${u}`;
  }
  setTextRenderingMode(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("textRenderingMode", t), this.current.textRenderingMode = e;
  }
  setTextRise(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("textRise", t), this.current.textRise = e;
  }
  moveText(t, e, n) {
    var s;
    (s = this.dependencyTracker) == null || s.resetIncrementalData("sameLineText").recordIncrementalData("moveText", t), this.current.x = this.current.lineX += e, this.current.y = this.current.lineY += n;
  }
  setLeadingMoveText(t, e, n) {
    this.setLeading(t, -n), this.moveText(t, e, n);
  }
  setTextMatrix(t, e) {
    var s;
    (s = this.dependencyTracker) == null || s.recordSimpleData("textMatrix", t);
    const {
      current: n
    } = this;
    n.textMatrix = e, n.textMatrixScale = Math.hypot(e[0], e[1]), n.x = n.lineX = 0, n.y = n.lineY = 0;
  }
  nextLine(t) {
    var e;
    this.moveText(t, 0, this.current.leading), (e = this.dependencyTracker) == null || e.recordIncrementalData("moveText", this.dependencyTracker.getSimpleIndex("leading") ?? t);
  }
  paintChar(t, e, n, s, a, l) {
    var T, R, P, L;
    const u = this.ctx, h = this.current, f = h.font, g = h.textRenderingMode, y = h.fontSize / h.fontSizeScale, v = g & dn.FILL_STROKE_MASK, S = !!(g & dn.ADD_TO_PATH_FLAG), x = h.patternFill && !f.missingFile, _ = h.patternStroke && !f.missingFile;
    let k;
    if ((f.disableFontFace || S || x || _) && !f.missingFile && (k = f.getPathGenerator(this.commonObjs, e)), k && (f.disableFontFace || x || _)) {
      u.save(), u.translate(n, s), u.scale(y, -y), (T = this.dependencyTracker) == null || T.recordCharacterBBox(t, u, f);
      let D;
      if (v === dn.FILL || v === dn.FILL_STROKE)
        if (a) {
          D = u.getTransform(), u.setTransform(...a);
          const N = E(this, Ps, dw).call(this, k, D, a);
          u.fill(N);
        } else
          u.fill(k);
      if (v === dn.STROKE || v === dn.FILL_STROKE)
        if (l) {
          D || (D = u.getTransform()), u.setTransform(...l);
          const {
            a: N,
            b: B,
            c: $,
            d: V
          } = D, X = ft.inverseTransform(l), q = ft.transform([N, B, $, V, 0, 0], X);
          ft.singularValueDecompose2dScale(q, Pi), u.lineWidth *= Math.max(Pi[0], Pi[1]) / y, u.stroke(E(this, Ps, dw).call(this, k, D, l));
        } else
          u.lineWidth /= y, u.stroke(k);
      u.restore();
    } else
      (v === dn.FILL || v === dn.FILL_STROKE) && (u.fillText(e, n, s), (R = this.dependencyTracker) == null || R.recordCharacterBBox(t, u, f, y, n, s, () => u.measureText(e))), (v === dn.STROKE || v === dn.FILL_STROKE) && (this.dependencyTracker && ((P = this.dependencyTracker) == null || P.recordCharacterBBox(t, u, f, y, n, s, () => u.measureText(e)).recordDependencies(t, ji.stroke)), u.strokeText(e, n, s));
    S && ((this.pendingTextPaths || (this.pendingTextPaths = [])).push({
      transform: _e(u),
      x: n,
      y: s,
      fontSize: y,
      path: k
    }), (L = this.dependencyTracker) == null || L.recordCharacterBBox(t, u, f, y, n, s));
  }
  get isFontSubpixelAAEnabled() {
    const {
      context: t
    } = this.cachedCanvases.getCanvas("isFontSubpixelAAEnabled", 10, 10);
    t.scale(1.5, 1), t.fillText("I", 0, 10);
    const e = t.getImageData(0, 0, 10, 10).data;
    let n = !1;
    for (let s = 3; s < e.length; s += 4)
      if (e[s] > 0 && e[s] < 255) {
        n = !0;
        break;
      }
    return It(this, "isFontSubpixelAAEnabled", n);
  }
  showText(t, e) {
    var $, V, X, q;
    this.dependencyTracker && (this.dependencyTracker.recordDependencies(t, ji.showText).resetBBox(t), this.current.textRenderingMode & dn.ADD_TO_PATH_FLAG && this.dependencyTracker.recordFutureForcedDependency("textClip", t).inheritPendingDependenciesAsFutureForcedDependencies());
    const n = this.current, s = n.font;
    if (s.isType3Font) {
      this.showType3Text(t, e), ($ = this.dependencyTracker) == null || $.recordShowTextOperation(t);
      return;
    }
    const a = n.fontSize;
    if (a === 0) {
      (V = this.dependencyTracker) == null || V.recordOperation(t);
      return;
    }
    const l = this.ctx, u = n.fontSizeScale, h = n.charSpacing, f = n.wordSpacing, g = n.fontDirection, y = n.textHScale * g, v = e.length, S = s.vertical, x = S ? 1 : -1, _ = s.defaultVMetrics, k = a * n.fontMatrix[0], T = n.textRenderingMode === dn.FILL && !s.disableFontFace && !n.patternFill;
    l.save(), n.textMatrix && l.transform(...n.textMatrix), l.translate(n.x, n.y + n.textRise), g > 0 ? l.scale(y, -1) : l.scale(y, 1);
    let R, P;
    if (n.patternFill) {
      l.save();
      const it = n.fillColor.getPattern(l, this, Ns(l), vn.FILL, t);
      R = _e(l), l.restore(), l.fillStyle = it;
    }
    if (n.patternStroke) {
      l.save();
      const it = n.strokeColor.getPattern(l, this, Ns(l), vn.STROKE, t);
      P = _e(l), l.restore(), l.strokeStyle = it;
    }
    let L = n.lineWidth;
    const D = n.textMatrixScale;
    if (D === 0 || L === 0) {
      const it = n.textRenderingMode & dn.FILL_STROKE_MASK;
      (it === dn.STROKE || it === dn.FILL_STROKE) && (L = this.getSinglePixelWidth());
    } else
      L /= D;
    if (u !== 1 && (l.scale(u, u), L /= u), l.lineWidth = L, s.isInvalidPDFjsFont) {
      const it = [];
      let lt = 0;
      for (const vt of e)
        it.push(vt.unicode), lt += vt.width;
      const rt = it.join("");
      if (l.fillText(rt, 0, 0), this.dependencyTracker !== null) {
        const vt = l.measureText(rt);
        this.dependencyTracker.recordBBox(t, this.ctx, -vt.actualBoundingBoxLeft, vt.actualBoundingBoxRight, -vt.actualBoundingBoxAscent, vt.actualBoundingBoxDescent).recordShowTextOperation(t);
      }
      n.x += lt * k * y, l.restore(), this.compose();
      return;
    }
    let N = 0, B;
    for (B = 0; B < v; ++B) {
      const it = e[B];
      if (typeof it == "number") {
        N += x * it * a / 1e3;
        continue;
      }
      let lt = !1;
      const rt = (it.isSpace ? f : 0) + h, vt = it.fontChar, tt = it.accent;
      let Q, W, ot = it.width;
      if (S) {
        const z = it.vmetric || _, K = -(it.vmetric ? z[1] : ot * 0.5) * k, ct = z[2] * k;
        ot = z ? -z[0] : ot, Q = K / u, W = (N + ct) / u;
      } else
        Q = N / u, W = 0;
      let st;
      if (s.remeasure && ot > 0) {
        st = l.measureText(vt);
        const z = st.width * 1e3 / a * u;
        if (ot < z && this.isFontSubpixelAAEnabled) {
          const K = ot / z;
          lt = !0, l.save(), l.scale(K, 1), Q /= K;
        } else ot !== z && (Q += (ot - z) / 2e3 * a / u);
      }
      if (this.contentVisible && (it.isInFont || s.missingFile)) {
        if (T && !tt)
          l.fillText(vt, Q, W), (X = this.dependencyTracker) == null || X.recordCharacterBBox(t, l, st ? {
            bbox: null
          } : s, a / u, Q, W, () => st ?? l.measureText(vt));
        else if (this.paintChar(t, vt, Q, W, R, P), tt) {
          const z = Q + a * tt.offset.x / u, K = W - a * tt.offset.y / u;
          this.paintChar(t, tt.fontChar, z, K, R, P);
        }
      }
      const I = S ? ot * k - rt * g : ot * k + rt * g;
      N += I, lt && l.restore();
    }
    S ? n.y -= N : n.x += N * y, l.restore(), this.compose(), (q = this.dependencyTracker) == null || q.recordShowTextOperation(t);
  }
  showType3Text(t, e) {
    const n = this.ctx, s = this.current, a = s.font, l = s.fontSize, u = s.fontDirection, h = a.vertical ? 1 : -1, f = s.charSpacing, g = s.wordSpacing, y = s.textHScale * u, v = s.fontMatrix || F0, S = e.length, x = s.textRenderingMode === dn.INVISIBLE;
    let _, k, T, R;
    if (x || l === 0)
      return;
    this._cachedScaleForStroking[0] = -1, this._cachedGetSinglePixelWidth = null, n.save(), s.textMatrix && n.transform(...s.textMatrix), n.translate(s.x, s.y + s.textRise), n.scale(y, u);
    const P = this.dependencyTracker;
    for (this.dependencyTracker = P ? new Gm(P, t) : null, _ = 0; _ < S; ++_) {
      if (k = e[_], typeof k == "number") {
        R = h * k * l / 1e3, this.ctx.translate(R, 0), s.x += R * y;
        continue;
      }
      const L = (k.isSpace ? g : 0) + f, D = a.charProcOperatorList[k.operatorListId];
      D ? this.contentVisible && (this.save(), n.scale(l, l), n.transform(...v), this.executeOperatorList(D), this.restore()) : Rt(`Type3 character "${k.operatorListId}" is not available.`);
      const N = [k.width, 0];
      ft.applyTransform(N, v), T = N[0] * l + L, n.translate(T, 0), s.x += T * y;
    }
    n.restore(), P && (this.dependencyTracker = P);
  }
  setCharWidth(t, e, n) {
  }
  setCharWidthAndBounds(t, e, n, s, a, l, u) {
    var f;
    const h = new Path2D();
    h.rect(s, a, l - s, u - a), this.ctx.clip(h), (f = this.dependencyTracker) == null || f.recordBBox(t, this.ctx, s, l, a, u).recordClipBox(t, this.ctx, s, l, a, u), this.endPath(t);
  }
  getColorN_Pattern(t, e) {
    let n;
    if (e[0] === "TilingPattern") {
      const s = this.baseTransform || _e(this.ctx), a = {
        createCanvasGraphics: (l, u) => new TA(l, this.commonObjs, this.objs, this.canvasFactory, this.filterFactory, {
          optionalContentConfig: this.optionalContentConfig,
          markedContentStack: this.markedContentStack
        }, void 0, void 0, this.dependencyTracker ? new Gm(this.dependencyTracker, u, !0) : null)
      };
      n = new cw(e, this.ctx, a, s);
    } else
      n = this._getPattern(t, e[1], e[2]);
    return n;
  }
  setStrokeColorN(t, ...e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("strokeColor", t), this.current.strokeColor = this.getColorN_Pattern(t, e), this.current.patternStroke = !0;
  }
  setFillColorN(t, ...e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("fillColor", t), this.current.fillColor = this.getColorN_Pattern(t, e), this.current.patternFill = !0;
  }
  setStrokeRGBColor(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("strokeColor", t), this.ctx.strokeStyle = this.current.strokeColor = e, this.current.patternStroke = !1;
  }
  setStrokeTransparent(t) {
    var e;
    (e = this.dependencyTracker) == null || e.recordSimpleData("strokeColor", t), this.ctx.strokeStyle = this.current.strokeColor = "transparent", this.current.patternStroke = !1;
  }
  setFillRGBColor(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.recordSimpleData("fillColor", t), this.ctx.fillStyle = this.current.fillColor = e, this.current.patternFill = !1;
  }
  setFillTransparent(t) {
    var e;
    (e = this.dependencyTracker) == null || e.recordSimpleData("fillColor", t), this.ctx.fillStyle = this.current.fillColor = "transparent", this.current.patternFill = !1;
  }
  _getPattern(t, e, n = null) {
    let s;
    return this.cachedPatterns.has(e) ? s = this.cachedPatterns.get(e) : (s = Vk(this.getObject(t, e)), this.cachedPatterns.set(e, s)), n && (s.matrix = n), s;
  }
  shadingFill(t, e) {
    var l;
    if (!this.contentVisible)
      return;
    const n = this.ctx;
    this.save(t);
    const s = this._getPattern(t, e);
    n.fillStyle = s.getPattern(n, this, Ns(n), vn.SHADING, t);
    const a = Ns(n);
    if (a) {
      const {
        width: u,
        height: h
      } = n.canvas, f = Yc.slice();
      ft.axialAlignedBoundingBox([0, 0, u, h], a, f);
      const [g, y, v, S] = f;
      this.ctx.fillRect(g, y, v - g, S - y);
    } else
      this.ctx.fillRect(-1e10, -1e10, 2e10, 2e10);
    (l = this.dependencyTracker) == null || l.resetBBox(t).recordFullPageBBox(t).recordDependencies(t, ji.transform).recordDependencies(t, ji.fill).recordOperation(t), this.compose(this.current.getClippedPathBoundingBox()), this.restore(t);
  }
  beginInlineImage() {
    ce("Should not call beginInlineImage");
  }
  beginImageData() {
    ce("Should not call beginImageData");
  }
  paintFormXObjectBegin(t, e, n) {
    var s;
    if (this.contentVisible && (this.save(t), this.baseTransformStack.push(this.baseTransform), e && this.transform(t, ...e), this.baseTransform = _e(this.ctx), n)) {
      ft.axialAlignedBoundingBox(n, this.baseTransform, this.current.minMax);
      const [a, l, u, h] = n, f = new Path2D();
      f.rect(a, l, u - a, h - l), this.ctx.clip(f), (s = this.dependencyTracker) == null || s.recordClipBox(t, this.ctx, a, u, l, h), this.endPath(t);
    }
  }
  paintFormXObjectEnd(t) {
    this.contentVisible && (this.restore(t), this.baseTransform = this.baseTransformStack.pop());
  }
  beginGroup(t, e) {
    var P;
    if (!this.contentVisible)
      return;
    this.save(t), this.inSMaskMode && (this.endSMaskMode(), this.current.activeSMask = null);
    const n = this.ctx;
    e.isolated || Sy("TODO: Support non-isolated groups."), e.knockout && Rt("Knockout groups not supported.");
    const s = _e(n);
    if (e.matrix && n.transform(...e.matrix), !e.bbox)
      throw new Error("Bounding box is required.");
    let a = Yc.slice();
    ft.axialAlignedBoundingBox(e.bbox, _e(n), a);
    const l = [0, 0, n.canvas.width, n.canvas.height];
    a = ft.intersect(a, l) || [0, 0, 0, 0];
    const u = Math.floor(a[0]), h = Math.floor(a[1]), f = Math.max(Math.ceil(a[2]) - u, 1), g = Math.max(Math.ceil(a[3]) - h, 1);
    this.current.startNewPathAndClipBox([0, 0, f, g]);
    let y = "groupAt" + this.groupLevel;
    e.smask && (y += "_smask_" + this.smaskCounter++ % 2);
    const v = this.cachedCanvases.getCanvas(y, f, g), S = v.context;
    S.translate(-u, -h), S.transform(...s);
    let x = new Path2D();
    const [_, k, T, R] = e.bbox;
    if (x.rect(_, k, T - _, R - k), e.matrix) {
      const L = new Path2D();
      L.addPath(x, new DOMMatrix(e.matrix)), x = L;
    }
    S.clip(x), e.smask && this.smaskStack.push({
      canvas: v.canvas,
      context: S,
      offsetX: u,
      offsetY: h,
      subtype: e.smask.subtype,
      backdrop: e.smask.backdrop,
      transferMap: e.smask.transferMap || null,
      startTransformInverse: null
    }), (!e.smask || this.dependencyTracker) && (n.setTransform(1, 0, 0, 1, 0, 0), n.translate(u, h), n.save()), wd(n, S), this.ctx = S, (P = this.dependencyTracker) == null || P.inheritSimpleDataAsFutureForcedDependencies(["fillAlpha", "strokeAlpha", "globalCompositeOperation"]).pushBaseTransform(n), this.setGState(t, [["BM", "source-over"], ["ca", 1], ["CA", 1]]), this.groupStack.push(n), this.groupLevel++;
  }
  endGroup(t, e) {
    var a;
    if (!this.contentVisible)
      return;
    this.groupLevel--;
    const n = this.ctx, s = this.groupStack.pop();
    if (this.ctx = s, this.ctx.imageSmoothingEnabled = !1, (a = this.dependencyTracker) == null || a.popBaseTransform(), e.smask)
      this.tempSMask = this.smaskStack.pop(), this.restore(t), this.dependencyTracker && this.ctx.restore();
    else {
      this.ctx.restore();
      const l = _e(this.ctx);
      this.restore(t), this.ctx.save(), this.ctx.setTransform(...l);
      const u = Yc.slice();
      ft.axialAlignedBoundingBox([0, 0, n.canvas.width, n.canvas.height], l, u), this.ctx.drawImage(n.canvas, 0, 0), this.ctx.restore(), this.compose(u);
    }
  }
  beginAnnotation(t, e, n, s, a, l) {
    if (E(this, Ps, uw).call(this), am(this.ctx), this.ctx.save(), this.save(t), this.baseTransform && this.ctx.setTransform(...this.baseTransform), n) {
      const u = n[2] - n[0], h = n[3] - n[1];
      if (l && this.annotationCanvasMap) {
        s = s.slice(), s[4] -= n[0], s[5] -= n[1], n = n.slice(), n[0] = n[1] = 0, n[2] = u, n[3] = h, ft.singularValueDecompose2dScale(_e(this.ctx), Pi);
        const {
          viewportScale: f
        } = this, g = Math.ceil(u * this.outputScaleX * f), y = Math.ceil(h * this.outputScaleY * f);
        this.annotationCanvas = this.canvasFactory.create(g, y);
        const {
          canvas: v,
          context: S
        } = this.annotationCanvas;
        this.annotationCanvasMap.set(e, v), this.annotationCanvas.savedCtx = this.ctx, this.ctx = S, this.ctx.save(), this.ctx.setTransform(Pi[0], 0, 0, -Pi[1], 0, h * Pi[1]), am(this.ctx);
      } else {
        am(this.ctx), this.endPath(t);
        const f = new Path2D();
        f.rect(n[0], n[1], u, h), this.ctx.clip(f);
      }
    }
    this.current = new XE(this.ctx.canvas.width, this.ctx.canvas.height), this.transform(t, ...s), this.transform(t, ...a);
  }
  endAnnotation(t) {
    this.annotationCanvas && (this.ctx.restore(), E(this, Ps, hw).call(this), this.ctx = this.annotationCanvas.savedCtx, delete this.annotationCanvas.savedCtx, delete this.annotationCanvas);
  }
  paintImageMaskXObject(t, e) {
    var u;
    if (!this.contentVisible)
      return;
    const n = e.count;
    e = this.getObject(t, e.data, e), e.count = n;
    const s = this.ctx, a = this._createMaskCanvas(t, e), l = a.canvas;
    s.save(), s.setTransform(1, 0, 0, 1, 0, 0), s.drawImage(l, a.offsetX, a.offsetY), (u = this.dependencyTracker) == null || u.resetBBox(t).recordBBox(t, this.ctx, a.offsetX, a.offsetX + l.width, a.offsetY, a.offsetY + l.height).recordOperation(t), s.restore(), this.compose();
  }
  paintImageMaskXObjectRepeat(t, e, n, s = 0, a = 0, l, u) {
    var y, v, S;
    if (!this.contentVisible)
      return;
    e = this.getObject(t, e.data, e);
    const h = this.ctx;
    h.save();
    const f = _e(h);
    h.transform(n, s, a, l, 0, 0);
    const g = this._createMaskCanvas(t, e);
    h.setTransform(1, 0, 0, 1, g.offsetX - f[4], g.offsetY - f[5]), (y = this.dependencyTracker) == null || y.resetBBox(t);
    for (let x = 0, _ = u.length; x < _; x += 2) {
      const k = ft.transform(f, [n, s, a, l, u[x], u[x + 1]]);
      h.drawImage(g.canvas, k[4], k[5]), (v = this.dependencyTracker) == null || v.recordBBox(t, this.ctx, k[4], k[4] + g.canvas.width, k[5], k[5] + g.canvas.height);
    }
    h.restore(), this.compose(), (S = this.dependencyTracker) == null || S.recordOperation(t);
  }
  paintImageMaskXObjectGroup(t, e) {
    var l, u, h;
    if (!this.contentVisible)
      return;
    const n = this.ctx, s = this.current.fillColor, a = this.current.patternFill;
    (l = this.dependencyTracker) == null || l.resetBBox(t).recordDependencies(t, ji.transformAndFill);
    for (const f of e) {
      const {
        data: g,
        width: y,
        height: v,
        transform: S
      } = f, x = this.cachedCanvases.getCanvas("maskCanvas", y, v), _ = x.context;
      _.save();
      const k = this.getObject(t, g, f);
      YE(_, k), _.globalCompositeOperation = "source-in", _.fillStyle = a ? s.getPattern(_, this, Ns(n), vn.FILL, t) : s, _.fillRect(0, 0, y, v), _.restore(), n.save(), n.transform(...S), n.scale(1, -1), om(n, x.canvas, 0, 0, y, v, 0, -1, 1, 1), (u = this.dependencyTracker) == null || u.recordBBox(t, n, 0, y, 0, v), n.restore();
    }
    this.compose(), (h = this.dependencyTracker) == null || h.recordOperation(t);
  }
  paintImageXObject(t, e) {
    if (!this.contentVisible)
      return;
    const n = this.getObject(t, e);
    if (!n) {
      Rt("Dependent image isn't ready yet");
      return;
    }
    this.paintInlineImageXObject(t, n);
  }
  paintImageXObjectRepeat(t, e, n, s, a) {
    if (!this.contentVisible)
      return;
    const l = this.getObject(t, e);
    if (!l) {
      Rt("Dependent image isn't ready yet");
      return;
    }
    const u = l.width, h = l.height, f = [];
    for (let g = 0, y = a.length; g < y; g += 2)
      f.push({
        transform: [n, 0, 0, s, a[g], a[g + 1]],
        x: 0,
        y: 0,
        w: u,
        h
      });
    this.paintInlineImageXObjectGroup(t, l, f);
  }
  applyTransferMapsToCanvas(t) {
    return this.current.transferMaps !== "none" && (t.filter = this.current.transferMaps, t.drawImage(t.canvas, 0, 0), t.filter = "none"), t.canvas;
  }
  applyTransferMapsToBitmap(t) {
    if (this.current.transferMaps === "none")
      return t.bitmap;
    const {
      bitmap: e,
      width: n,
      height: s
    } = t, a = this.cachedCanvases.getCanvas("inlineImage", n, s), l = a.context;
    return l.filter = this.current.transferMaps, l.drawImage(e, 0, 0), l.filter = "none", a.canvas;
  }
  paintInlineImageXObject(t, e) {
    var f;
    if (!this.contentVisible)
      return;
    const n = e.width, s = e.height, a = this.ctx;
    this.save(t);
    const {
      filter: l
    } = a;
    l !== "none" && l !== "" && (a.filter = "none"), a.scale(1 / n, -1 / s);
    let u;
    if (e.bitmap)
      u = this.applyTransferMapsToBitmap(e);
    else if (typeof HTMLElement == "function" && e instanceof HTMLElement || !e.data)
      u = e;
    else {
      const y = this.cachedCanvases.getCanvas("inlineImage", n, s).context;
      qE(y, e), u = this.applyTransferMapsToCanvas(y);
    }
    const h = this._scaleImage(u, Ns(a));
    a.imageSmoothingEnabled = KE(_e(a), e.interpolate), (f = this.dependencyTracker) == null || f.resetBBox(t).recordBBox(t, a, 0, n, -s, 0).recordDependencies(t, ji.imageXObject).recordOperation(t), om(a, h.img, 0, 0, h.paintWidth, h.paintHeight, 0, -s, n, s), this.compose(), this.restore(t);
  }
  paintInlineImageXObjectGroup(t, e, n) {
    var l, u, h;
    if (!this.contentVisible)
      return;
    const s = this.ctx;
    let a;
    if (e.bitmap)
      a = e.bitmap;
    else {
      const f = e.width, g = e.height, v = this.cachedCanvases.getCanvas("inlineImage", f, g).context;
      qE(v, e), a = this.applyTransferMapsToCanvas(v);
    }
    (l = this.dependencyTracker) == null || l.resetBBox(t);
    for (const f of n)
      s.save(), s.transform(...f.transform), s.scale(1, -1), om(s, a, f.x, f.y, f.w, f.h, 0, -1, 1, 1), (u = this.dependencyTracker) == null || u.recordBBox(t, s, 0, 1, -1, 0), s.restore();
    (h = this.dependencyTracker) == null || h.recordOperation(t), this.compose();
  }
  paintSolidColorImageMask(t) {
    var e;
    this.contentVisible && ((e = this.dependencyTracker) == null || e.resetBBox(t).recordBBox(t, this.ctx, 0, 1, 0, 1).recordDependencies(t, ji.fill).recordOperation(t), this.ctx.fillRect(0, 0, 1, 1), this.compose());
  }
  markPoint(t, e) {
  }
  markPointProps(t, e, n) {
  }
  beginMarkedContent(t, e) {
    var n;
    (n = this.dependencyTracker) == null || n.beginMarkedContent(t), this.markedContentStack.push({
      visible: !0
    });
  }
  beginMarkedContentProps(t, e, n) {
    var s;
    (s = this.dependencyTracker) == null || s.beginMarkedContent(t), e === "OC" ? this.markedContentStack.push({
      visible: this.optionalContentConfig.isVisible(n)
    }) : this.markedContentStack.push({
      visible: !0
    }), this.contentVisible = this.isContentVisible();
  }
  endMarkedContent(t) {
    var e;
    (e = this.dependencyTracker) == null || e.endMarkedContent(t), this.markedContentStack.pop(), this.contentVisible = this.isContentVisible();
  }
  beginCompat(t) {
  }
  endCompat(t) {
  }
  consumePath(t, e, n) {
    var l, u;
    const s = this.current.isEmptyClip();
    this.pendingClip && this.current.updateClipFromPath(), this.pendingClip || this.compose(n);
    const a = this.ctx;
    this.pendingClip ? (s || (this.pendingClip === QE ? a.clip(e, "evenodd") : a.clip(e)), this.pendingClip = null, (l = this.dependencyTracker) == null || l.bboxToClipBoxDropOperation(t).recordFutureForcedDependency("clipPath", t)) : (u = this.dependencyTracker) == null || u.recordOperation(t), this.current.startNewPathAndClipBox(this.current.clipBox);
  }
  getSinglePixelWidth() {
    if (!this._cachedGetSinglePixelWidth) {
      const t = _e(this.ctx);
      if (t[1] === 0 && t[2] === 0)
        this._cachedGetSinglePixelWidth = 1 / Math.min(Math.abs(t[0]), Math.abs(t[3]));
      else {
        const e = Math.abs(t[0] * t[3] - t[2] * t[1]), n = Math.hypot(t[0], t[2]), s = Math.hypot(t[1], t[3]);
        this._cachedGetSinglePixelWidth = Math.max(n, s) / e;
      }
    }
    return this._cachedGetSinglePixelWidth;
  }
  getScaleForStroking() {
    if (this._cachedScaleForStroking[0] === -1) {
      const {
        lineWidth: t
      } = this.current, {
        a: e,
        b: n,
        c: s,
        d: a
      } = this.ctx.getTransform();
      let l, u;
      if (n === 0 && s === 0) {
        const h = Math.abs(e), f = Math.abs(a);
        if (h === f)
          if (t === 0)
            l = u = 1 / h;
          else {
            const g = h * t;
            l = u = g < 1 ? 1 / g : 1;
          }
        else if (t === 0)
          l = 1 / h, u = 1 / f;
        else {
          const g = h * t, y = f * t;
          l = g < 1 ? 1 / g : 1, u = y < 1 ? 1 / y : 1;
        }
      } else {
        const h = Math.abs(e * a - n * s), f = Math.hypot(e, n), g = Math.hypot(s, a);
        if (t === 0)
          l = g / h, u = f / h;
        else {
          const y = t * h;
          l = g > y ? g / y : 1, u = f > y ? f / y : 1;
        }
      }
      this._cachedScaleForStroking[0] = l, this._cachedScaleForStroking[1] = u;
    }
    return this._cachedScaleForStroking;
  }
  rescaleAndStroke(t, e) {
    const {
      ctx: n,
      current: {
        lineWidth: s
      }
    } = this, [a, l] = this.getScaleForStroking();
    if (a === l) {
      n.lineWidth = (s || 1) * a, n.stroke(t);
      return;
    }
    const u = n.getLineDash();
    e && n.save(), n.scale(a, l), P0.a = 1 / a, P0.d = 1 / l;
    const h = new Path2D();
    if (h.addPath(t, P0), u.length > 0) {
      const f = Math.max(a, l);
      n.setLineDash(u.map((g) => g / f)), n.lineDashOffset /= f;
    }
    n.lineWidth = s || 1, n.stroke(h), e && n.restore();
  }
  isContentVisible() {
    for (let t = this.markedContentStack.length - 1; t >= 0; t--)
      if (!this.markedContentStack[t].visible)
        return !1;
    return !0;
  }
};
Ps = new WeakSet(), uw = function() {
  for (; this.stateStack.length || this.inSMaskMode; )
    this.restore();
  this.current.activeSMask = null, this.ctx.restore(), this.transparentCanvas && (this.ctx = this.compositeCtx, this.ctx.save(), this.ctx.setTransform(1, 0, 0, 1, 0, 0), this.ctx.drawImage(this.transparentCanvas, 0, 0), this.ctx.restore(), this.transparentCanvas = null);
}, hw = function() {
  if (this.pageColors) {
    const t = this.filterFactory.addHCMFilter(this.pageColors.foreground, this.pageColors.background);
    if (t !== "none") {
      const e = this.ctx.filter;
      this.ctx.filter = t, this.ctx.drawImage(this.ctx.canvas, 0, 0), this.ctx.filter = e;
    }
  }
}, dw = function(t, e, n) {
  const s = new Path2D();
  return s.addPath(t, new DOMMatrix(n).invertSelf().multiplySelf(e)), s;
};
let tu = TA;
for (const p in Ph)
  tu.prototype[p] !== void 0 && (tu.prototype[Ph[p]] = tu.prototype[p]);
var Ru, Mu, Mf, Lu, wm;
const Qc = class Qc {
  constructor(t) {
    b(this, Lu);
    b(this, Ru);
    b(this, Mu);
    b(this, Mf);
    w(this, Ru, t), w(this, Mu, new DataView(o(this, Ru))), w(this, Mf, new TextDecoder());
  }
  static write(t) {
    const e = new TextEncoder(), n = {};
    let s = 0;
    for (const f of Qc.strings) {
      const g = e.encode(t[f]);
      n[f] = g, s += 4 + g.length;
    }
    const a = new ArrayBuffer(s), l = new Uint8Array(a), u = new DataView(a);
    let h = 0;
    for (const f of Qc.strings) {
      const g = n[f], y = g.length;
      u.setUint32(h, y), l.set(g, h + 4), h += 4 + y;
    }
    return $t(h === a.byteLength, "CssFontInfo.write: Buffer overflow"), a;
  }
  get fontFamily() {
    return E(this, Lu, wm).call(this, 0);
  }
  get fontWeight() {
    return E(this, Lu, wm).call(this, 1);
  }
  get italicAngle() {
    return E(this, Lu, wm).call(this, 2);
  }
};
Ru = new WeakMap(), Mu = new WeakMap(), Mf = new WeakMap(), Lu = new WeakSet(), wm = function(t) {
  $t(t < Qc.strings.length, "Invalid string index");
  let e = 0;
  for (let s = 0; s < t; s++)
    e += o(this, Mu).getUint32(e) + 4;
  const n = o(this, Mu).getUint32(e);
  return o(this, Mf).decode(new Uint8Array(o(this, Ru), e + 4, n));
}, Z(Qc, "strings", ["fontFamily", "fontWeight", "italicAngle"]);
let Xm = Qc;
var Zo, Ks, Cl, xl, Id;
const Zc = class Zc {
  constructor(t) {
    b(this, xl);
    b(this, Zo);
    b(this, Ks);
    b(this, Cl);
    w(this, Zo, t), w(this, Ks, new DataView(o(this, Zo))), w(this, Cl, new TextDecoder());
  }
  static write(t) {
    const e = new TextEncoder(), n = {};
    let s = 0;
    for (const v of Zc.strings) {
      const S = e.encode(t[v]);
      n[v] = S, s += 4 + S.length;
    }
    s += 4;
    let a, l, u = 1 + s;
    t.style && (a = e.encode(t.style.style), l = e.encode(t.style.weight), u += 4 + a.length + 4 + l.length);
    const h = new ArrayBuffer(u), f = new Uint8Array(h), g = new DataView(h);
    let y = 0;
    g.setUint8(y++, t.guessFallback ? 1 : 0), g.setUint32(y, 0), y += 4, s = 0;
    for (const v of Zc.strings) {
      const S = n[v], x = S.length;
      s += 4 + x, g.setUint32(y, x), f.set(S, y + 4), y += 4 + x;
    }
    return g.setUint32(y - s - 4, s), t.style && (g.setUint32(y, a.length), f.set(a, y + 4), y += 4 + a.length, g.setUint32(y, l.length), f.set(l, y + 4), y += 4 + l.length), $t(y <= h.byteLength, "SubstitionInfo.write: Buffer overflow"), h.transferToFixedLength(y);
  }
  get guessFallback() {
    return o(this, Ks).getUint8(0) !== 0;
  }
  get css() {
    return E(this, xl, Id).call(this, 0);
  }
  get loadedName() {
    return E(this, xl, Id).call(this, 1);
  }
  get baseFontName() {
    return E(this, xl, Id).call(this, 2);
  }
  get src() {
    return E(this, xl, Id).call(this, 3);
  }
  get style() {
    let t = 1;
    t += 4 + o(this, Ks).getUint32(t);
    const e = o(this, Ks).getUint32(t), n = o(this, Cl).decode(new Uint8Array(o(this, Zo), t + 4, e));
    t += 4 + e;
    const s = o(this, Ks).getUint32(t), a = o(this, Cl).decode(new Uint8Array(o(this, Zo), t + 4, s));
    return {
      style: n,
      weight: a
    };
  }
};
Zo = new WeakMap(), Ks = new WeakMap(), Cl = new WeakMap(), xl = new WeakSet(), Id = function(t) {
  $t(t < Zc.strings.length, "Invalid string index");
  let e = 5;
  for (let s = 0; s < t; s++)
    e += o(this, Ks).getUint32(e) + 4;
  const n = o(this, Ks).getUint32(e);
  return o(this, Cl).decode(new Uint8Array(o(this, Zo), e + 4, n));
}, Z(Zc, "strings", ["css", "loadedName", "baseFontName", "src"]);
let qm = Zc;
var Du, Iu, Fu, Nu, pi, Qs, Lf, ne, De, fs, Am, Fd;
const Lt = class Lt {
  constructor({
    data: t,
    extra: e
  }) {
    b(this, De);
    b(this, Qs);
    b(this, Lf);
    b(this, ne);
    w(this, Qs, t), w(this, Lf, new TextDecoder()), w(this, ne, new DataView(o(this, Qs))), e && Object.assign(this, e);
  }
  get black() {
    return E(this, De, fs).call(this, 0);
  }
  get bold() {
    return E(this, De, fs).call(this, 1);
  }
  get disableFontFace() {
    return E(this, De, fs).call(this, 2);
  }
  get fontExtraProperties() {
    return E(this, De, fs).call(this, 3);
  }
  get isInvalidPDFjsFont() {
    return E(this, De, fs).call(this, 4);
  }
  get isType3Font() {
    return E(this, De, fs).call(this, 5);
  }
  get italic() {
    return E(this, De, fs).call(this, 6);
  }
  get missingFile() {
    return E(this, De, fs).call(this, 7);
  }
  get remeasure() {
    return E(this, De, fs).call(this, 8);
  }
  get vertical() {
    return E(this, De, fs).call(this, 9);
  }
  get ascent() {
    return E(this, De, Am).call(this, 0);
  }
  get defaultWidth() {
    return E(this, De, Am).call(this, 1);
  }
  get descent() {
    return E(this, De, Am).call(this, 2);
  }
  get bbox() {
    let t = o(Lt, Iu);
    if (o(this, ne).getUint8(t) === 0)
      return;
    t += 1;
    const n = [];
    for (let s = 0; s < 4; s++)
      n.push(o(this, ne).getInt16(t, !0)), t += 2;
    return n;
  }
  get fontMatrix() {
    let t = o(Lt, Fu);
    if (o(this, ne).getUint8(t) === 0)
      return;
    t += 1;
    const n = [];
    for (let s = 0; s < 6; s++)
      n.push(o(this, ne).getFloat64(t, !0)), t += 8;
    return n;
  }
  get defaultVMetrics() {
    let t = o(Lt, Nu);
    if (o(this, ne).getUint8(t) === 0)
      return;
    t += 1;
    const n = [];
    for (let s = 0; s < 3; s++)
      n.push(o(this, ne).getInt16(t, !0)), t += 2;
    return n;
  }
  get fallbackName() {
    return E(this, De, Fd).call(this, 0);
  }
  get loadedName() {
    return E(this, De, Fd).call(this, 1);
  }
  get mimetype() {
    return E(this, De, Fd).call(this, 2);
  }
  get name() {
    return E(this, De, Fd).call(this, 3);
  }
  get data() {
    let t = o(Lt, pi);
    const e = o(this, ne).getUint32(t);
    t += 4 + e;
    const n = o(this, ne).getUint32(t);
    t += 4 + n;
    const s = o(this, ne).getUint32(t);
    t += 4 + s;
    const a = o(this, ne).getUint32(t);
    if (a !== 0)
      return new Uint8Array(o(this, Qs), t + 4, a);
  }
  clearData() {
    let t = o(Lt, pi);
    const e = o(this, ne).getUint32(t);
    t += 4 + e;
    const n = o(this, ne).getUint32(t);
    t += 4 + n;
    const s = o(this, ne).getUint32(t);
    t += 4 + s;
    const a = o(this, ne).getUint32(t);
    new Uint8Array(o(this, Qs), t + 4, a).fill(0), o(this, ne).setUint32(t, 0);
  }
  get cssFontInfo() {
    let t = o(Lt, pi);
    const e = o(this, ne).getUint32(t);
    t += 4 + e;
    const n = o(this, ne).getUint32(t);
    t += 4 + n;
    const s = o(this, ne).getUint32(t);
    if (s === 0)
      return null;
    const a = new Uint8Array(s);
    return a.set(new Uint8Array(o(this, Qs), t + 4, s)), new Xm(a.buffer);
  }
  get systemFontInfo() {
    let t = o(Lt, pi);
    const e = o(this, ne).getUint32(t);
    t += 4 + e;
    const n = o(this, ne).getUint32(t);
    if (n === 0)
      return null;
    const s = new Uint8Array(n);
    return s.set(new Uint8Array(o(this, Qs), t + 4, n)), new qm(s.buffer);
  }
  static write(t) {
    const e = t.systemFontInfo ? qm.write(t.systemFontInfo) : null, n = t.cssFontInfo ? Xm.write(t.cssFontInfo) : null, s = new TextEncoder(), a = {};
    let l = 0;
    for (const _ of Lt.strings)
      a[_] = s.encode(t[_]), l += 4 + a[_].length;
    const u = o(Lt, pi) + 4 + l + 4 + (e ? e.byteLength : 0) + 4 + (n ? n.byteLength : 0) + 4 + (t.data ? t.data.length : 0), h = new ArrayBuffer(u), f = new Uint8Array(h), g = new DataView(h);
    let y = 0;
    const v = Lt.bools.length;
    let S = 0, x = 0;
    for (let _ = 0; _ < v; _++) {
      const k = t[Lt.bools[_]];
      S |= (k === void 0 ? 0 : k ? 2 : 1) << x, x += 2, (x === 8 || _ === v - 1) && (g.setUint8(y++, S), S = 0, x = 0);
    }
    $t(y === o(Lt, Du), "FontInfo.write: Boolean properties offset mismatch");
    for (const _ of Lt.numbers)
      g.setFloat64(y, t[_]), y += 8;
    if ($t(y === o(Lt, Iu), "FontInfo.write: Number properties offset mismatch"), t.bbox) {
      g.setUint8(y++, 4);
      for (const _ of t.bbox)
        g.setInt16(y, _, !0), y += 2;
    } else
      g.setUint8(y++, 0), y += 2 * 4;
    if ($t(y === o(Lt, Fu), "FontInfo.write: BBox properties offset mismatch"), t.fontMatrix) {
      g.setUint8(y++, 6);
      for (const _ of t.fontMatrix)
        g.setFloat64(y, _, !0), y += 8;
    } else
      g.setUint8(y++, 0), y += 8 * 6;
    if ($t(y === o(Lt, Nu), "FontInfo.write: FontMatrix properties offset mismatch"), t.defaultVMetrics) {
      g.setUint8(y++, 1);
      for (const _ of t.defaultVMetrics)
        g.setInt16(y, _, !0), y += 2;
    } else
      g.setUint8(y++, 0), y += 3 * 2;
    $t(y === o(Lt, pi), "FontInfo.write: DefaultVMetrics properties offset mismatch"), g.setUint32(o(Lt, pi), 0), y += 4;
    for (const _ of Lt.strings) {
      const k = a[_], T = k.length;
      g.setUint32(y, T), f.set(k, y + 4), y += 4 + T;
    }
    if (g.setUint32(o(Lt, pi), y - o(Lt, pi) - 4), !e)
      g.setUint32(y, 0), y += 4;
    else {
      const _ = e.byteLength;
      g.setUint32(y, _), $t(y + 4 + _ <= h.byteLength, "FontInfo.write: Buffer overflow at systemFontInfo"), f.set(new Uint8Array(e), y + 4), y += 4 + _;
    }
    if (!n)
      g.setUint32(y, 0), y += 4;
    else {
      const _ = n.byteLength;
      g.setUint32(y, _), $t(y + 4 + _ <= h.byteLength, "FontInfo.write: Buffer overflow at cssFontInfo"), f.set(new Uint8Array(n), y + 4), y += 4 + _;
    }
    return t.data === void 0 ? (g.setUint32(y, 0), y += 4) : (g.setUint32(y, t.data.length), f.set(t.data, y + 4), y += 4 + t.data.length), $t(y <= h.byteLength, "FontInfo.write: Buffer overflow"), h.transferToFixedLength(y);
  }
};
Du = new WeakMap(), Iu = new WeakMap(), Fu = new WeakMap(), Nu = new WeakMap(), pi = new WeakMap(), Qs = new WeakMap(), Lf = new WeakMap(), ne = new WeakMap(), De = new WeakSet(), fs = function(t) {
  $t(t < Lt.bools.length, "Invalid boolean index");
  const e = Math.floor(t / 4), n = t * 2 % 8, s = o(this, ne).getUint8(e) >> n & 3;
  return s === 0 ? void 0 : s === 2;
}, Am = function(t) {
  return $t(t < Lt.numbers.length, "Invalid number index"), o(this, ne).getFloat64(o(Lt, Du) + t * 8);
}, Fd = function(t) {
  $t(t < Lt.strings.length, "Invalid string index");
  let e = o(Lt, pi) + 4;
  for (let a = 0; a < t; a++)
    e += o(this, ne).getUint32(e) + 4;
  const n = o(this, ne).getUint32(e), s = new Uint8Array(n);
  return s.set(new Uint8Array(o(this, Qs), e + 4, n)), o(this, Lf).decode(s);
}, Z(Lt, "bools", ["black", "bold", "disableFontFace", "fontExtraProperties", "isInvalidPDFjsFont", "isType3Font", "italic", "missingFile", "remeasure", "vertical"]), Z(Lt, "numbers", ["ascent", "defaultWidth", "descent"]), Z(Lt, "strings", ["fallbackName", "loadedName", "mimetype", "name"]), b(Lt, Du, Math.ceil(Lt.bools.length * 2 / 8)), b(Lt, Iu, o(Lt, Du) + Lt.numbers.length * 8), b(Lt, Fu, o(Lt, Iu) + 1 + 2 * 4), b(Lt, Nu, o(Lt, Fu) + 1 + 8 * 6), b(Lt, pi, o(Lt, Nu) + 1 + 2 * 3);
let fw = Lt;
var Df, If;
class ts {
  static get workerPort() {
    return o(this, Df);
  }
  static set workerPort(t) {
    if (!(typeof Worker < "u" && t instanceof Worker) && t !== null)
      throw new Error("Invalid `workerPort` type.");
    w(this, Df, t);
  }
  static get workerSrc() {
    return o(this, If);
  }
  static set workerSrc(t) {
    if (typeof t != "string")
      throw new Error("Invalid `workerSrc` type.");
    w(this, If, t);
  }
}
Df = new WeakMap(), If = new WeakMap(), b(ts, Df, null), b(ts, If, "");
var Ou, Ff;
class Zk {
  constructor({
    parsedData: t,
    rawData: e
  }) {
    b(this, Ou);
    b(this, Ff);
    w(this, Ou, t), w(this, Ff, e);
  }
  getRaw() {
    return o(this, Ff);
  }
  get(t) {
    return o(this, Ou).get(t) ?? null;
  }
  [Symbol.iterator]() {
    return o(this, Ou).entries();
  }
}
Ou = new WeakMap(), Ff = new WeakMap();
const Gc = Symbol("INTERNAL");
var Nf, Of, Bf, Bu;
class Jk {
  constructor(t, {
    name: e,
    intent: n,
    usage: s,
    rbGroups: a
  }) {
    b(this, Nf, !1);
    b(this, Of, !1);
    b(this, Bf, !1);
    b(this, Bu, !0);
    w(this, Nf, !!(t & ki.DISPLAY)), w(this, Of, !!(t & ki.PRINT)), this.name = e, this.intent = n, this.usage = s, this.rbGroups = a;
  }
  get visible() {
    if (o(this, Bf))
      return o(this, Bu);
    if (!o(this, Bu))
      return !1;
    const {
      print: t,
      view: e
    } = this.usage;
    return o(this, Nf) ? (e == null ? void 0 : e.viewState) !== "OFF" : o(this, Of) ? (t == null ? void 0 : t.printState) !== "OFF" : !0;
  }
  _setVisible(t, e, n = !1) {
    t !== Gc && ce("Internal method `_setVisible` called."), w(this, Bf, n), w(this, Bu, e);
  }
}
Nf = new WeakMap(), Of = new WeakMap(), Bf = new WeakMap(), Bu = new WeakMap();
var Jo, ie, zu, Uu, zf, pw;
class tP {
  constructor(t, e = ki.DISPLAY) {
    b(this, zf);
    b(this, Jo, null);
    b(this, ie, /* @__PURE__ */ new Map());
    b(this, zu, null);
    b(this, Uu, null);
    if (this.renderingIntent = e, this.name = null, this.creator = null, t !== null) {
      this.name = t.name, this.creator = t.creator, w(this, Uu, t.order);
      for (const n of t.groups)
        o(this, ie).set(n.id, new Jk(e, n));
      if (t.baseState === "OFF")
        for (const n of o(this, ie).values())
          n._setVisible(Gc, !1);
      for (const n of t.on)
        o(this, ie).get(n)._setVisible(Gc, !0);
      for (const n of t.off)
        o(this, ie).get(n)._setVisible(Gc, !1);
      w(this, zu, this.getHash());
    }
  }
  isVisible(t) {
    if (o(this, ie).size === 0)
      return !0;
    if (!t)
      return Sy("Optional content group not defined."), !0;
    if (t.type === "OCG")
      return o(this, ie).has(t.id) ? o(this, ie).get(t.id).visible : (Rt(`Optional content group not found: ${t.id}`), !0);
    if (t.type === "OCMD") {
      if (t.expression)
        return E(this, zf, pw).call(this, t.expression);
      if (!t.policy || t.policy === "AnyOn") {
        for (const e of t.ids) {
          if (!o(this, ie).has(e))
            return Rt(`Optional content group not found: ${e}`), !0;
          if (o(this, ie).get(e).visible)
            return !0;
        }
        return !1;
      } else if (t.policy === "AllOn") {
        for (const e of t.ids) {
          if (!o(this, ie).has(e))
            return Rt(`Optional content group not found: ${e}`), !0;
          if (!o(this, ie).get(e).visible)
            return !1;
        }
        return !0;
      } else if (t.policy === "AnyOff") {
        for (const e of t.ids) {
          if (!o(this, ie).has(e))
            return Rt(`Optional content group not found: ${e}`), !0;
          if (!o(this, ie).get(e).visible)
            return !0;
        }
        return !1;
      } else if (t.policy === "AllOff") {
        for (const e of t.ids) {
          if (!o(this, ie).has(e))
            return Rt(`Optional content group not found: ${e}`), !0;
          if (o(this, ie).get(e).visible)
            return !1;
        }
        return !0;
      }
      return Rt(`Unknown optional content policy ${t.policy}.`), !0;
    }
    return Rt(`Unknown group type ${t.type}.`), !0;
  }
  setVisibility(t, e = !0, n = !0) {
    var a;
    const s = o(this, ie).get(t);
    if (!s) {
      Rt(`Optional content group not found: ${t}`);
      return;
    }
    if (n && e && s.rbGroups.length)
      for (const l of s.rbGroups)
        for (const u of l)
          u !== t && ((a = o(this, ie).get(u)) == null || a._setVisible(Gc, !1, !0));
    s._setVisible(Gc, !!e, !0), w(this, Jo, null);
  }
  setOCGState({
    state: t,
    preserveRB: e
  }) {
    let n;
    for (const s of t) {
      switch (s) {
        case "ON":
        case "OFF":
        case "Toggle":
          n = s;
          continue;
      }
      const a = o(this, ie).get(s);
      if (a)
        switch (n) {
          case "ON":
            this.setVisibility(s, !0, e);
            break;
          case "OFF":
            this.setVisibility(s, !1, e);
            break;
          case "Toggle":
            this.setVisibility(s, !a.visible, e);
            break;
        }
    }
    w(this, Jo, null);
  }
  get hasInitialVisibility() {
    return o(this, zu) === null || this.getHash() === o(this, zu);
  }
  getOrder() {
    return o(this, ie).size ? o(this, Uu) ? o(this, Uu).slice() : [...o(this, ie).keys()] : null;
  }
  getGroup(t) {
    return o(this, ie).get(t) || null;
  }
  getHash() {
    if (o(this, Jo) !== null)
      return o(this, Jo);
    const t = new s_();
    for (const [e, n] of o(this, ie))
      t.update(`${e}:${n.visible}`);
    return w(this, Jo, t.hexdigest());
  }
  [Symbol.iterator]() {
    return o(this, ie).entries();
  }
}
Jo = new WeakMap(), ie = new WeakMap(), zu = new WeakMap(), Uu = new WeakMap(), zf = new WeakSet(), pw = function(t) {
  const e = t.length;
  if (e < 2)
    return !0;
  const n = t[0];
  for (let s = 1; s < e; s++) {
    const a = t[s];
    let l;
    if (Array.isArray(a))
      l = E(this, zf, pw).call(this, a);
    else if (o(this, ie).has(a))
      l = o(this, ie).get(a).visible;
    else
      return Rt(`Optional content group not found: ${a}`), !0;
    switch (n) {
      case "And":
        if (!l)
          return !1;
        break;
      case "Or":
        if (l)
          return !0;
        break;
      case "Not":
        return !l;
      default:
        return !0;
    }
  }
  return n === "And";
};
class eP {
  constructor(t, {
    disableRange: e = !1,
    disableStream: n = !1
  }) {
    $t(t, 'PDFDataTransportStream - missing required "pdfDataRangeTransport" argument.');
    const {
      length: s,
      initialData: a,
      progressiveDone: l,
      contentDispositionFilename: u
    } = t;
    if (this._queuedChunks = [], this._progressiveDone = l, this._contentDispositionFilename = u, (a == null ? void 0 : a.length) > 0) {
      const h = a instanceof Uint8Array && a.byteLength === a.buffer.byteLength ? a.buffer : new Uint8Array(a).buffer;
      this._queuedChunks.push(h);
    }
    this._pdfDataRangeTransport = t, this._isStreamingSupported = !n, this._isRangeSupported = !e, this._contentLength = s, this._fullRequestReader = null, this._rangeReaders = [], t.addRangeListener((h, f) => {
      this._onReceiveData({
        begin: h,
        chunk: f
      });
    }), t.addProgressListener((h, f) => {
      this._onProgress({
        loaded: h,
        total: f
      });
    }), t.addProgressiveReadListener((h) => {
      this._onReceiveData({
        chunk: h
      });
    }), t.addProgressiveDoneListener(() => {
      this._onProgressiveDone();
    }), t.transportReady();
  }
  _onReceiveData({
    begin: t,
    chunk: e
  }) {
    const n = e instanceof Uint8Array && e.byteLength === e.buffer.byteLength ? e.buffer : new Uint8Array(e).buffer;
    if (t === void 0)
      this._fullRequestReader ? this._fullRequestReader._enqueue(n) : this._queuedChunks.push(n);
    else {
      const s = this._rangeReaders.some(function(a) {
        return a._begin !== t ? !1 : (a._enqueue(n), !0);
      });
      $t(s, "_onReceiveData - no `PDFDataTransportStreamRangeReader` instance found.");
    }
  }
  get _progressiveDataLength() {
    var t;
    return ((t = this._fullRequestReader) == null ? void 0 : t._loaded) ?? 0;
  }
  _onProgress(t) {
    var e, n, s, a;
    t.total === void 0 ? (n = (e = this._rangeReaders[0]) == null ? void 0 : e.onProgress) == null || n.call(e, {
      loaded: t.loaded
    }) : (a = (s = this._fullRequestReader) == null ? void 0 : s.onProgress) == null || a.call(s, {
      loaded: t.loaded,
      total: t.total
    });
  }
  _onProgressiveDone() {
    var t;
    (t = this._fullRequestReader) == null || t.progressiveDone(), this._progressiveDone = !0;
  }
  _removeRangeReader(t) {
    const e = this._rangeReaders.indexOf(t);
    e >= 0 && this._rangeReaders.splice(e, 1);
  }
  getFullReader() {
    $t(!this._fullRequestReader, "PDFDataTransportStream.getFullReader can only be called once.");
    const t = this._queuedChunks;
    return this._queuedChunks = null, new nP(this, t, this._progressiveDone, this._contentDispositionFilename);
  }
  getRangeReader(t, e) {
    if (e <= this._progressiveDataLength)
      return null;
    const n = new iP(this, t, e);
    return this._pdfDataRangeTransport.requestDataRange(t, e), this._rangeReaders.push(n), n;
  }
  cancelAllRequests(t) {
    var e;
    (e = this._fullRequestReader) == null || e.cancel(t);
    for (const n of this._rangeReaders.slice(0))
      n.cancel(t);
    this._pdfDataRangeTransport.abort();
  }
}
class nP {
  constructor(t, e, n = !1, s = null) {
    this._stream = t, this._done = n || !1, this._filename = Ey(s) ? s : null, this._queuedChunks = e || [], this._loaded = 0;
    for (const a of this._queuedChunks)
      this._loaded += a.byteLength;
    this._requests = [], this._headersReady = Promise.resolve(), t._fullRequestReader = this, this.onProgress = null;
  }
  _enqueue(t) {
    this._done || (this._requests.length > 0 ? this._requests.shift().resolve({
      value: t,
      done: !1
    }) : this._queuedChunks.push(t), this._loaded += t.byteLength);
  }
  get headersReady() {
    return this._headersReady;
  }
  get filename() {
    return this._filename;
  }
  get isRangeSupported() {
    return this._stream._isRangeSupported;
  }
  get isStreamingSupported() {
    return this._stream._isStreamingSupported;
  }
  get contentLength() {
    return this._stream._contentLength;
  }
  async read() {
    if (this._queuedChunks.length > 0)
      return {
        value: this._queuedChunks.shift(),
        done: !1
      };
    if (this._done)
      return {
        value: void 0,
        done: !0
      };
    const t = Promise.withResolvers();
    return this._requests.push(t), t.promise;
  }
  cancel(t) {
    this._done = !0;
    for (const e of this._requests)
      e.resolve({
        value: void 0,
        done: !0
      });
    this._requests.length = 0;
  }
  progressiveDone() {
    this._done || (this._done = !0);
  }
}
class iP {
  constructor(t, e, n) {
    this._stream = t, this._begin = e, this._end = n, this._queuedChunk = null, this._requests = [], this._done = !1, this.onProgress = null;
  }
  _enqueue(t) {
    if (!this._done) {
      if (this._requests.length === 0)
        this._queuedChunk = t;
      else {
        this._requests.shift().resolve({
          value: t,
          done: !1
        });
        for (const n of this._requests)
          n.resolve({
            value: void 0,
            done: !0
          });
        this._requests.length = 0;
      }
      this._done = !0, this._stream._removeRangeReader(this);
    }
  }
  get isStreamingSupported() {
    return !1;
  }
  async read() {
    if (this._queuedChunk) {
      const e = this._queuedChunk;
      return this._queuedChunk = null, {
        value: e,
        done: !1
      };
    }
    if (this._done)
      return {
        value: void 0,
        done: !0
      };
    const t = Promise.withResolvers();
    return this._requests.push(t), t.promise;
  }
  cancel(t) {
    this._done = !0;
    for (const e of this._requests)
      e.resolve({
        value: void 0,
        done: !0
      });
    this._requests.length = 0, this._stream._removeRangeReader(this);
  }
}
function sP(p) {
  let t = !0, e = n("filename\\*", "i").exec(p);
  if (e) {
    e = e[1];
    let g = u(e);
    return g = unescape(g), g = h(g), g = f(g), a(g);
  }
  if (e = l(p), e) {
    const g = f(e);
    return a(g);
  }
  if (e = n("filename", "i").exec(p), e) {
    e = e[1];
    let g = u(e);
    return g = f(g), a(g);
  }
  function n(g, y) {
    return new RegExp("(?:^|;)\\s*" + g + '\\s*=\\s*([^";\\s][^;\\s]*|"(?:[^"\\\\]|\\\\"?)+"?)', y);
  }
  function s(g, y) {
    if (g) {
      if (!/^[\x00-\xFF]+$/.test(y))
        return y;
      try {
        const v = new TextDecoder(g, {
          fatal: !0
        }), S = Fp(y);
        y = v.decode(S), t = !1;
      } catch {
      }
    }
    return y;
  }
  function a(g) {
    return t && /[\x80-\xff]/.test(g) && (g = s("utf-8", g), t && (g = s("iso-8859-1", g))), g;
  }
  function l(g) {
    const y = [];
    let v;
    const S = n("filename\\*((?!0\\d)\\d+)(\\*?)", "ig");
    for (; (v = S.exec(g)) !== null; ) {
      let [, _, k, T] = v;
      if (_ = parseInt(_, 10), _ in y) {
        if (_ === 0)
          break;
        continue;
      }
      y[_] = [k, T];
    }
    const x = [];
    for (let _ = 0; _ < y.length && _ in y; ++_) {
      let [k, T] = y[_];
      T = u(T), k && (T = unescape(T), _ === 0 && (T = h(T))), x.push(T);
    }
    return x.join("");
  }
  function u(g) {
    if (g.startsWith('"')) {
      const y = g.slice(1).split('\\"');
      for (let v = 0; v < y.length; ++v) {
        const S = y[v].indexOf('"');
        S !== -1 && (y[v] = y[v].slice(0, S), y.length = v + 1), y[v] = y[v].replaceAll(/\\(.)/g, "$1");
      }
      g = y.join('"');
    }
    return g;
  }
  function h(g) {
    const y = g.indexOf("'");
    if (y === -1)
      return g;
    const v = g.slice(0, y), x = g.slice(y + 1).replace(/^[^']*'/, "");
    return s(v, x);
  }
  function f(g) {
    return !g.startsWith("=?") || /[\x00-\x19\x80-\xff]/.test(g) ? g : g.replaceAll(/=\?([\w-]*)\?([QqBb])\?((?:[^?]|\?(?!=))*)\?=/g, function(y, v, S, x) {
      if (S === "q" || S === "Q")
        return x = x.replaceAll("_", " "), x = x.replaceAll(/=([0-9a-fA-F]{2})/g, function(_, k) {
          return String.fromCharCode(parseInt(k, 16));
        }), s(v, x);
      try {
        x = atob(x);
      } catch {
      }
      return s(v, x);
    });
  }
  return "";
}
function y_(p, t) {
  const e = new Headers();
  if (!p || !t || typeof t != "object")
    return e;
  for (const n in t) {
    const s = t[n];
    s !== void 0 && e.append(n, s);
  }
  return e;
}
function _y(p) {
  var t;
  return ((t = URL.parse(p)) == null ? void 0 : t.origin) ?? null;
}
function v_({
  responseHeaders: p,
  isHttp: t,
  rangeChunkSize: e,
  disableRange: n
}) {
  const s = {
    allowRangeRequests: !1,
    suggestedLength: void 0
  }, a = parseInt(p.get("Content-Length"), 10);
  return !Number.isInteger(a) || (s.suggestedLength = a, a <= 2 * e) || n || !t || p.get("Accept-Ranges") !== "bytes" || (p.get("Content-Encoding") || "identity") !== "identity" || (s.allowRangeRequests = !0), s;
}
function w_(p) {
  const t = p.get("Content-Disposition");
  if (t) {
    let e = sP(t);
    if (e.includes("%"))
      try {
        e = decodeURIComponent(e);
      } catch {
      }
    if (Ey(e))
      return e;
  }
  return null;
}
function zp(p, t) {
  return new Gd(`Unexpected server response (${p}) while retrieving PDF "${t}".`, p, p === 404 || p === 0 && t.startsWith("file:"));
}
function A_(p) {
  return p === 200 || p === 206;
}
function S_(p, t, e) {
  return {
    method: "GET",
    headers: p,
    signal: e.signal,
    mode: "cors",
    credentials: t ? "include" : "same-origin",
    redirect: "follow"
  };
}
function b_(p) {
  return p instanceof Uint8Array ? p.buffer : p instanceof ArrayBuffer ? p : (Rt(`getArrayBuffer - unexpected data format: ${p}`), new Uint8Array(p).buffer);
}
class rP {
  constructor(t) {
    Z(this, "_responseOrigin", null);
    this.source = t, this.isHttp = /^https?:/i.test(t.url), this.headers = y_(this.isHttp, t.httpHeaders), this._fullRequestReader = null, this._rangeRequestReaders = [];
  }
  get _progressiveDataLength() {
    var t;
    return ((t = this._fullRequestReader) == null ? void 0 : t._loaded) ?? 0;
  }
  getFullReader() {
    return $t(!this._fullRequestReader, "PDFFetchStream.getFullReader can only be called once."), this._fullRequestReader = new oP(this), this._fullRequestReader;
  }
  getRangeReader(t, e) {
    if (e <= this._progressiveDataLength)
      return null;
    const n = new aP(this, t, e);
    return this._rangeRequestReaders.push(n), n;
  }
  cancelAllRequests(t) {
    var e;
    (e = this._fullRequestReader) == null || e.cancel(t);
    for (const n of this._rangeRequestReaders.slice(0))
      n.cancel(t);
  }
}
class oP {
  constructor(t) {
    this._stream = t, this._reader = null, this._loaded = 0, this._filename = null;
    const e = t.source;
    this._withCredentials = e.withCredentials || !1, this._contentLength = e.length, this._headersCapability = Promise.withResolvers(), this._disableRange = e.disableRange || !1, this._rangeChunkSize = e.rangeChunkSize, !this._rangeChunkSize && !this._disableRange && (this._disableRange = !0), this._abortController = new AbortController(), this._isStreamingSupported = !e.disableStream, this._isRangeSupported = !e.disableRange;
    const n = new Headers(t.headers), s = e.url;
    fetch(s, S_(n, this._withCredentials, this._abortController)).then((a) => {
      if (t._responseOrigin = _y(a.url), !A_(a.status))
        throw zp(a.status, s);
      this._reader = a.body.getReader(), this._headersCapability.resolve();
      const l = a.headers, {
        allowRangeRequests: u,
        suggestedLength: h
      } = v_({
        responseHeaders: l,
        isHttp: t.isHttp,
        rangeChunkSize: this._rangeChunkSize,
        disableRange: this._disableRange
      });
      this._isRangeSupported = u, this._contentLength = h || this._contentLength, this._filename = w_(l), !this._isStreamingSupported && this._isRangeSupported && this.cancel(new ao("Streaming is disabled."));
    }).catch(this._headersCapability.reject), this.onProgress = null;
  }
  get headersReady() {
    return this._headersCapability.promise;
  }
  get filename() {
    return this._filename;
  }
  get contentLength() {
    return this._contentLength;
  }
  get isRangeSupported() {
    return this._isRangeSupported;
  }
  get isStreamingSupported() {
    return this._isStreamingSupported;
  }
  async read() {
    var n;
    await this._headersCapability.promise;
    const {
      value: t,
      done: e
    } = await this._reader.read();
    return e ? {
      value: t,
      done: e
    } : (this._loaded += t.byteLength, (n = this.onProgress) == null || n.call(this, {
      loaded: this._loaded,
      total: this._contentLength
    }), {
      value: b_(t),
      done: !1
    });
  }
  cancel(t) {
    var e;
    (e = this._reader) == null || e.cancel(t), this._abortController.abort();
  }
}
class aP {
  constructor(t, e, n) {
    this._stream = t, this._reader = null, this._loaded = 0;
    const s = t.source;
    this._withCredentials = s.withCredentials || !1, this._readCapability = Promise.withResolvers(), this._isStreamingSupported = !s.disableStream, this._abortController = new AbortController();
    const a = new Headers(t.headers);
    a.append("Range", `bytes=${e}-${n - 1}`);
    const l = s.url;
    fetch(l, S_(a, this._withCredentials, this._abortController)).then((u) => {
      const h = _y(u.url);
      if (h !== t._responseOrigin)
        throw new Error(`Expected range response-origin "${h}" to match "${t._responseOrigin}".`);
      if (!A_(u.status))
        throw zp(u.status, l);
      this._readCapability.resolve(), this._reader = u.body.getReader();
    }).catch(this._readCapability.reject), this.onProgress = null;
  }
  get isStreamingSupported() {
    return this._isStreamingSupported;
  }
  async read() {
    var n;
    await this._readCapability.promise;
    const {
      value: t,
      done: e
    } = await this._reader.read();
    return e ? {
      value: t,
      done: e
    } : (this._loaded += t.byteLength, (n = this.onProgress) == null || n.call(this, {
      loaded: this._loaded
    }), {
      value: b_(t),
      done: !1
    });
  }
  cancel(t) {
    var e;
    (e = this._reader) == null || e.cancel(t), this._abortController.abort();
  }
}
const R0 = 200, M0 = 206;
function lP(p) {
  const t = p.response;
  return typeof t != "string" ? t : Fp(t).buffer;
}
class cP {
  constructor({
    url: t,
    httpHeaders: e,
    withCredentials: n
  }) {
    Z(this, "_responseOrigin", null);
    this.url = t, this.isHttp = /^https?:/i.test(t), this.headers = y_(this.isHttp, e), this.withCredentials = n || !1, this.currXhrId = 0, this.pendingRequests = /* @__PURE__ */ Object.create(null);
  }
  request(t) {
    const e = new XMLHttpRequest(), n = this.currXhrId++, s = this.pendingRequests[n] = {
      xhr: e
    };
    e.open("GET", this.url), e.withCredentials = this.withCredentials;
    for (const [a, l] of this.headers)
      e.setRequestHeader(a, l);
    return this.isHttp && "begin" in t && "end" in t ? (e.setRequestHeader("Range", `bytes=${t.begin}-${t.end - 1}`), s.expectedStatus = M0) : s.expectedStatus = R0, e.responseType = "arraybuffer", $t(t.onError, "Expected `onError` callback to be provided."), e.onerror = () => {
      t.onError(e.status);
    }, e.onreadystatechange = this.onStateChange.bind(this, n), e.onprogress = this.onProgress.bind(this, n), s.onHeadersReceived = t.onHeadersReceived, s.onDone = t.onDone, s.onError = t.onError, s.onProgress = t.onProgress, e.send(null), n;
  }
  onProgress(t, e) {
    var s;
    const n = this.pendingRequests[t];
    n && ((s = n.onProgress) == null || s.call(n, e));
  }
  onStateChange(t, e) {
    const n = this.pendingRequests[t];
    if (!n)
      return;
    const s = n.xhr;
    if (s.readyState >= 2 && n.onHeadersReceived && (n.onHeadersReceived(), delete n.onHeadersReceived), s.readyState !== 4 || !(t in this.pendingRequests))
      return;
    if (delete this.pendingRequests[t], s.status === 0 && this.isHttp) {
      n.onError(s.status);
      return;
    }
    const a = s.status || R0;
    if (!(a === R0 && n.expectedStatus === M0) && a !== n.expectedStatus) {
      n.onError(s.status);
      return;
    }
    const u = lP(s);
    if (a === M0) {
      const h = s.getResponseHeader("Content-Range"), f = /bytes (\d+)-(\d+)\/(\d+)/.exec(h);
      f ? n.onDone({
        begin: parseInt(f[1], 10),
        chunk: u
      }) : (Rt('Missing or invalid "Content-Range" header.'), n.onError(0));
    } else u ? n.onDone({
      begin: 0,
      chunk: u
    }) : n.onError(s.status);
  }
  getRequestXhr(t) {
    return this.pendingRequests[t].xhr;
  }
  isPendingRequest(t) {
    return t in this.pendingRequests;
  }
  abortRequest(t) {
    const e = this.pendingRequests[t].xhr;
    delete this.pendingRequests[t], e.abort();
  }
}
class uP {
  constructor(t) {
    this._source = t, this._manager = new cP(t), this._rangeChunkSize = t.rangeChunkSize, this._fullRequestReader = null, this._rangeRequestReaders = [];
  }
  _onRangeRequestReaderClosed(t) {
    const e = this._rangeRequestReaders.indexOf(t);
    e >= 0 && this._rangeRequestReaders.splice(e, 1);
  }
  getFullReader() {
    return $t(!this._fullRequestReader, "PDFNetworkStream.getFullReader can only be called once."), this._fullRequestReader = new hP(this._manager, this._source), this._fullRequestReader;
  }
  getRangeReader(t, e) {
    const n = new dP(this._manager, t, e);
    return n.onClosed = this._onRangeRequestReaderClosed.bind(this), this._rangeRequestReaders.push(n), n;
  }
  cancelAllRequests(t) {
    var e;
    (e = this._fullRequestReader) == null || e.cancel(t);
    for (const n of this._rangeRequestReaders.slice(0))
      n.cancel(t);
  }
}
class hP {
  constructor(t, e) {
    this._manager = t, this._url = e.url, this._fullRequestId = t.request({
      onHeadersReceived: this._onHeadersReceived.bind(this),
      onDone: this._onDone.bind(this),
      onError: this._onError.bind(this),
      onProgress: this._onProgress.bind(this)
    }), this._headersCapability = Promise.withResolvers(), this._disableRange = e.disableRange || !1, this._contentLength = e.length, this._rangeChunkSize = e.rangeChunkSize, !this._rangeChunkSize && !this._disableRange && (this._disableRange = !0), this._isStreamingSupported = !1, this._isRangeSupported = !1, this._cachedChunks = [], this._requests = [], this._done = !1, this._storedError = void 0, this._filename = null, this.onProgress = null;
  }
  _onHeadersReceived() {
    const t = this._fullRequestId, e = this._manager.getRequestXhr(t);
    this._manager._responseOrigin = _y(e.responseURL);
    const n = e.getAllResponseHeaders(), s = new Headers(n ? n.trimStart().replace(/[^\S ]+$/, "").split(/[\r\n]+/).map((u) => {
      const [h, ...f] = u.split(": ");
      return [h, f.join(": ")];
    }) : []), {
      allowRangeRequests: a,
      suggestedLength: l
    } = v_({
      responseHeaders: s,
      isHttp: this._manager.isHttp,
      rangeChunkSize: this._rangeChunkSize,
      disableRange: this._disableRange
    });
    a && (this._isRangeSupported = !0), this._contentLength = l || this._contentLength, this._filename = w_(s), this._isRangeSupported && this._manager.abortRequest(t), this._headersCapability.resolve();
  }
  _onDone(t) {
    if (t && (this._requests.length > 0 ? this._requests.shift().resolve({
      value: t.chunk,
      done: !1
    }) : this._cachedChunks.push(t.chunk)), this._done = !0, !(this._cachedChunks.length > 0)) {
      for (const e of this._requests)
        e.resolve({
          value: void 0,
          done: !0
        });
      this._requests.length = 0;
    }
  }
  _onError(t) {
    this._storedError = zp(t, this._url), this._headersCapability.reject(this._storedError);
    for (const e of this._requests)
      e.reject(this._storedError);
    this._requests.length = 0, this._cachedChunks.length = 0;
  }
  _onProgress(t) {
    var e;
    (e = this.onProgress) == null || e.call(this, {
      loaded: t.loaded,
      total: t.lengthComputable ? t.total : this._contentLength
    });
  }
  get filename() {
    return this._filename;
  }
  get isRangeSupported() {
    return this._isRangeSupported;
  }
  get isStreamingSupported() {
    return this._isStreamingSupported;
  }
  get contentLength() {
    return this._contentLength;
  }
  get headersReady() {
    return this._headersCapability.promise;
  }
  async read() {
    if (await this._headersCapability.promise, this._storedError)
      throw this._storedError;
    if (this._cachedChunks.length > 0)
      return {
        value: this._cachedChunks.shift(),
        done: !1
      };
    if (this._done)
      return {
        value: void 0,
        done: !0
      };
    const t = Promise.withResolvers();
    return this._requests.push(t), t.promise;
  }
  cancel(t) {
    this._done = !0, this._headersCapability.reject(t);
    for (const e of this._requests)
      e.resolve({
        value: void 0,
        done: !0
      });
    this._requests.length = 0, this._manager.isPendingRequest(this._fullRequestId) && this._manager.abortRequest(this._fullRequestId), this._fullRequestReader = null;
  }
}
class dP {
  constructor(t, e, n) {
    this._manager = t, this._url = t.url, this._requestId = t.request({
      begin: e,
      end: n,
      onHeadersReceived: this._onHeadersReceived.bind(this),
      onDone: this._onDone.bind(this),
      onError: this._onError.bind(this),
      onProgress: this._onProgress.bind(this)
    }), this._requests = [], this._queuedChunk = null, this._done = !1, this._storedError = void 0, this.onProgress = null, this.onClosed = null;
  }
  _onHeadersReceived() {
    var e;
    const t = _y((e = this._manager.getRequestXhr(this._requestId)) == null ? void 0 : e.responseURL);
    t !== this._manager._responseOrigin && (this._storedError = new Error(`Expected range response-origin "${t}" to match "${this._manager._responseOrigin}".`), this._onError(0));
  }
  _close() {
    var t;
    (t = this.onClosed) == null || t.call(this, this);
  }
  _onDone(t) {
    const e = t.chunk;
    this._requests.length > 0 ? this._requests.shift().resolve({
      value: e,
      done: !1
    }) : this._queuedChunk = e, this._done = !0;
    for (const n of this._requests)
      n.resolve({
        value: void 0,
        done: !0
      });
    this._requests.length = 0, this._close();
  }
  _onError(t) {
    this._storedError ?? (this._storedError = zp(t, this._url));
    for (const e of this._requests)
      e.reject(this._storedError);
    this._requests.length = 0, this._queuedChunk = null;
  }
  _onProgress(t) {
    var e;
    this.isStreamingSupported || (e = this.onProgress) == null || e.call(this, {
      loaded: t.loaded
    });
  }
  get isStreamingSupported() {
    return !1;
  }
  async read() {
    if (this._storedError)
      throw this._storedError;
    if (this._queuedChunk !== null) {
      const e = this._queuedChunk;
      return this._queuedChunk = null, {
        value: e,
        done: !1
      };
    }
    if (this._done)
      return {
        value: void 0,
        done: !0
      };
    const t = Promise.withResolvers();
    return this._requests.push(t), t.promise;
  }
  cancel(t) {
    this._done = !0;
    for (const e of this._requests)
      e.resolve({
        value: void 0,
        done: !0
      });
    this._requests.length = 0, this._manager.isPendingRequest(this._requestId) && this._manager.abortRequest(this._requestId), this._close();
  }
}
const fP = /^[a-z][a-z0-9\-+.]+:/i;
function pP(p) {
  if (fP.test(p))
    return new URL(p);
  const t = process.getBuiltinModule("url");
  return new URL(t.pathToFileURL(p));
}
class gP {
  constructor(t) {
    this.source = t, this.url = pP(t.url), $t(this.url.protocol === "file:", "PDFNodeStream only supports file:// URLs."), this._fullRequestReader = null, this._rangeRequestReaders = [];
  }
  get _progressiveDataLength() {
    var t;
    return ((t = this._fullRequestReader) == null ? void 0 : t._loaded) ?? 0;
  }
  getFullReader() {
    return $t(!this._fullRequestReader, "PDFNodeStream.getFullReader can only be called once."), this._fullRequestReader = new mP(this), this._fullRequestReader;
  }
  getRangeReader(t, e) {
    if (e <= this._progressiveDataLength)
      return null;
    const n = new yP(this, t, e);
    return this._rangeRequestReaders.push(n), n;
  }
  cancelAllRequests(t) {
    var e;
    (e = this._fullRequestReader) == null || e.cancel(t);
    for (const n of this._rangeRequestReaders.slice(0))
      n.cancel(t);
  }
}
class mP {
  constructor(t) {
    this._url = t.url, this._done = !1, this._storedError = null, this.onProgress = null;
    const e = t.source;
    this._contentLength = e.length, this._loaded = 0, this._filename = null, this._disableRange = e.disableRange || !1, this._rangeChunkSize = e.rangeChunkSize, !this._rangeChunkSize && !this._disableRange && (this._disableRange = !0), this._isStreamingSupported = !e.disableStream, this._isRangeSupported = !e.disableRange, this._readableStream = null, this._readCapability = Promise.withResolvers(), this._headersCapability = Promise.withResolvers();
    const n = process.getBuiltinModule("fs");
    n.promises.lstat(this._url).then((s) => {
      this._contentLength = s.size, this._setReadableStream(n.createReadStream(this._url)), this._headersCapability.resolve();
    }, (s) => {
      s.code === "ENOENT" && (s = zp(0, this._url.href)), this._storedError = s, this._headersCapability.reject(s);
    });
  }
  get headersReady() {
    return this._headersCapability.promise;
  }
  get filename() {
    return this._filename;
  }
  get contentLength() {
    return this._contentLength;
  }
  get isRangeSupported() {
    return this._isRangeSupported;
  }
  get isStreamingSupported() {
    return this._isStreamingSupported;
  }
  async read() {
    var n;
    if (await this._readCapability.promise, this._done)
      return {
        value: void 0,
        done: !0
      };
    if (this._storedError)
      throw this._storedError;
    const t = this._readableStream.read();
    return t === null ? (this._readCapability = Promise.withResolvers(), this.read()) : (this._loaded += t.length, (n = this.onProgress) == null || n.call(this, {
      loaded: this._loaded,
      total: this._contentLength
    }), {
      value: new Uint8Array(t).buffer,
      done: !1
    });
  }
  cancel(t) {
    if (!this._readableStream) {
      this._error(t);
      return;
    }
    this._readableStream.destroy(t);
  }
  _error(t) {
    this._storedError = t, this._readCapability.resolve();
  }
  _setReadableStream(t) {
    this._readableStream = t, t.on("readable", () => {
      this._readCapability.resolve();
    }), t.on("end", () => {
      t.destroy(), this._done = !0, this._readCapability.resolve();
    }), t.on("error", (e) => {
      this._error(e);
    }), !this._isStreamingSupported && this._isRangeSupported && this._error(new ao("streaming is disabled")), this._storedError && this._readableStream.destroy(this._storedError);
  }
}
class yP {
  constructor(t, e, n) {
    this._url = t.url, this._done = !1, this._storedError = null, this.onProgress = null, this._loaded = 0, this._readableStream = null, this._readCapability = Promise.withResolvers();
    const s = t.source;
    this._isStreamingSupported = !s.disableStream;
    const a = process.getBuiltinModule("fs");
    this._setReadableStream(a.createReadStream(this._url, {
      start: e,
      end: n - 1
    }));
  }
  get isStreamingSupported() {
    return this._isStreamingSupported;
  }
  async read() {
    var n;
    if (await this._readCapability.promise, this._done)
      return {
        value: void 0,
        done: !0
      };
    if (this._storedError)
      throw this._storedError;
    const t = this._readableStream.read();
    return t === null ? (this._readCapability = Promise.withResolvers(), this.read()) : (this._loaded += t.length, (n = this.onProgress) == null || n.call(this, {
      loaded: this._loaded
    }), {
      value: new Uint8Array(t).buffer,
      done: !1
    });
  }
  cancel(t) {
    if (!this._readableStream) {
      this._error(t);
      return;
    }
    this._readableStream.destroy(t);
  }
  _error(t) {
    this._storedError = t, this._readCapability.resolve();
  }
  _setReadableStream(t) {
    this._readableStream = t, t.on("readable", () => {
      this._readCapability.resolve();
    }), t.on("end", () => {
      t.destroy(), this._done = !0, this._readCapability.resolve();
    }), t.on("error", (e) => {
      this._error(e);
    }), this._storedError && this._readableStream.destroy(this._storedError);
  }
}
const Ad = Symbol("INITIAL_DATA");
var gi, Uf, gw;
class E_ {
  constructor() {
    b(this, Uf);
    b(this, gi, /* @__PURE__ */ Object.create(null));
  }
  get(t, e = null) {
    if (e) {
      const s = E(this, Uf, gw).call(this, t);
      return s.promise.then(() => e(s.data)), null;
    }
    const n = o(this, gi)[t];
    if (!n || n.data === Ad)
      throw new Error(`Requesting object that isn't resolved yet ${t}.`);
    return n.data;
  }
  has(t) {
    const e = o(this, gi)[t];
    return !!e && e.data !== Ad;
  }
  delete(t) {
    const e = o(this, gi)[t];
    return !e || e.data === Ad ? !1 : (delete o(this, gi)[t], !0);
  }
  resolve(t, e = null) {
    const n = E(this, Uf, gw).call(this, t);
    n.data = e, n.resolve();
  }
  clear() {
    var t;
    for (const e in o(this, gi)) {
      const {
        data: n
      } = o(this, gi)[e];
      (t = n == null ? void 0 : n.bitmap) == null || t.close();
    }
    w(this, gi, /* @__PURE__ */ Object.create(null));
  }
  *[Symbol.iterator]() {
    for (const t in o(this, gi)) {
      const {
        data: e
      } = o(this, gi)[t];
      e !== Ad && (yield [t, e]);
    }
  }
}
gi = new WeakMap(), Uf = new WeakSet(), gw = function(t) {
  var e;
  return (e = o(this, gi))[t] || (e[t] = {
    ...Promise.withResolvers(),
    data: Ad
  });
};
const vP = 1e5, ZE = 30;
var p1, ta, ni, Hf, jf, Tl, jr, $f, Vf, kl, Hu, ju, ea, $u, Wf, Vu, Pl, Gf, Xf, Le, Wu, Rl, qf, na, Gu, uo, __, C_, mw, Mi, Sm, yw, x_, T_;
let Kd = (Le = class {
  constructor({
    textContentSource: t,
    container: e,
    viewport: n
  }) {
    b(this, uo);
    b(this, ta, Promise.withResolvers());
    b(this, ni, null);
    b(this, Hf, !1);
    b(this, jf, !!((p1 = globalThis.FontInspector) != null && p1.enabled));
    b(this, Tl, null);
    b(this, jr, null);
    b(this, $f, 0);
    b(this, Vf, 0);
    b(this, kl, null);
    b(this, Hu, null);
    b(this, ju, 0);
    b(this, ea, 0);
    b(this, $u, /* @__PURE__ */ Object.create(null));
    b(this, Wf, []);
    b(this, Vu, null);
    b(this, Pl, []);
    b(this, Gf, /* @__PURE__ */ new WeakMap());
    b(this, Xf, null);
    var h;
    if (t instanceof ReadableStream)
      w(this, Vu, t);
    else if (typeof t == "object")
      w(this, Vu, new ReadableStream({
        start(f) {
          f.enqueue(t), f.close();
        }
      }));
    else
      throw new Error('No "textContentSource" parameter specified.');
    w(this, ni, w(this, Hu, e)), w(this, ea, n.scale * ks.pixelRatio), w(this, ju, n.rotation), w(this, jr, {
      div: null,
      properties: null,
      ctx: null
    });
    const {
      pageWidth: s,
      pageHeight: a,
      pageX: l,
      pageY: u
    } = n.rawDims;
    w(this, Xf, [1, 0, 0, -1, -l, u + a]), w(this, Vf, s), w(this, $f, a), E(h = Le, Mi, x_).call(h), Ta(e, n), o(this, ta).promise.finally(() => {
      o(Le, Gu).delete(this), w(this, jr, null), w(this, $u, null);
    }).catch(() => {
    });
  }
  static get fontFamilyMap() {
    const {
      isWindows: t,
      isFirefox: e
    } = cn.platform;
    return It(this, "fontFamilyMap", /* @__PURE__ */ new Map([["sans-serif", `${t && e ? "Calibri, " : ""}sans-serif`], ["monospace", `${t && e ? "Lucida Console, " : ""}monospace`]]));
  }
  render() {
    const t = () => {
      o(this, kl).read().then(({
        value: e,
        done: n
      }) => {
        if (n) {
          o(this, ta).resolve();
          return;
        }
        o(this, Tl) ?? w(this, Tl, e.lang), Object.assign(o(this, $u), e.styles), E(this, uo, __).call(this, e.items), t();
      }, o(this, ta).reject);
    };
    return w(this, kl, o(this, Vu).getReader()), o(Le, Gu).add(this), t(), o(this, ta).promise;
  }
  update({
    viewport: t,
    onBefore: e = null
  }) {
    var a;
    const n = t.scale * ks.pixelRatio, s = t.rotation;
    if (s !== o(this, ju) && (e == null || e(), w(this, ju, s), Ta(o(this, Hu), {
      rotation: s
    })), n !== o(this, ea)) {
      e == null || e(), w(this, ea, n);
      const l = {
        div: null,
        properties: null,
        ctx: E(a = Le, Mi, Sm).call(a, o(this, Tl))
      };
      for (const u of o(this, Pl))
        l.properties = o(this, Gf).get(u), l.div = u, E(this, uo, mw).call(this, l);
    }
  }
  cancel() {
    var e;
    const t = new ao("TextLayer task cancelled.");
    (e = o(this, kl)) == null || e.cancel(t).catch(() => {
    }), w(this, kl, null), o(this, ta).reject(t);
  }
  get textDivs() {
    return o(this, Pl);
  }
  get textContentItemsStr() {
    return o(this, Wf);
  }
  static cleanup() {
    if (!(o(this, Gu).size > 0)) {
      o(this, Wu).clear();
      for (const {
        canvas: t
      } of o(this, Rl).values())
        t.remove();
      o(this, Rl).clear();
    }
  }
}, ta = new WeakMap(), ni = new WeakMap(), Hf = new WeakMap(), jf = new WeakMap(), Tl = new WeakMap(), jr = new WeakMap(), $f = new WeakMap(), Vf = new WeakMap(), kl = new WeakMap(), Hu = new WeakMap(), ju = new WeakMap(), ea = new WeakMap(), $u = new WeakMap(), Wf = new WeakMap(), Vu = new WeakMap(), Pl = new WeakMap(), Gf = new WeakMap(), Xf = new WeakMap(), Wu = new WeakMap(), Rl = new WeakMap(), qf = new WeakMap(), na = new WeakMap(), Gu = new WeakMap(), uo = new WeakSet(), __ = function(t) {
  var s, a;
  if (o(this, Hf))
    return;
  (a = o(this, jr)).ctx ?? (a.ctx = E(s = Le, Mi, Sm).call(s, o(this, Tl)));
  const e = o(this, Pl), n = o(this, Wf);
  for (const l of t) {
    if (e.length > vP) {
      Rt("Ignoring additional textDivs for performance reasons."), w(this, Hf, !0);
      return;
    }
    if (l.str === void 0) {
      if (l.type === "beginMarkedContentProps" || l.type === "beginMarkedContent") {
        const u = o(this, ni);
        w(this, ni, document.createElement("span")), o(this, ni).classList.add("markedContent"), l.id && o(this, ni).setAttribute("id", `${l.id}`), u.append(o(this, ni));
      } else l.type === "endMarkedContent" && w(this, ni, o(this, ni).parentNode);
      continue;
    }
    n.push(l.str), E(this, uo, C_).call(this, l);
  }
}, C_ = function(t) {
  var _;
  const e = document.createElement("span"), n = {
    angle: 0,
    canvasWidth: 0,
    hasText: t.str !== "",
    hasEOL: t.hasEOL,
    fontSize: 0
  };
  o(this, Pl).push(e);
  const s = ft.transform(o(this, Xf), t.transform);
  let a = Math.atan2(s[1], s[0]);
  const l = o(this, $u)[t.fontName];
  l.vertical && (a += Math.PI / 2);
  let u = o(this, jf) && l.fontSubstitution || l.fontFamily;
  u = Le.fontFamilyMap.get(u) || u;
  const h = Math.hypot(s[2], s[3]), f = h * E(_ = Le, Mi, T_).call(_, u, l, o(this, Tl));
  let g, y;
  a === 0 ? (g = s[4], y = s[5] - f) : (g = s[4] + f * Math.sin(a), y = s[5] - f * Math.cos(a));
  const v = "calc(var(--total-scale-factor) *", S = e.style;
  o(this, ni) === o(this, Hu) ? (S.left = `${(100 * g / o(this, Vf)).toFixed(2)}%`, S.top = `${(100 * y / o(this, $f)).toFixed(2)}%`) : (S.left = `${v}${g.toFixed(2)}px)`, S.top = `${v}${y.toFixed(2)}px)`), S.fontSize = `${v}${(o(Le, na) * h).toFixed(2)}px)`, S.fontFamily = u, n.fontSize = h, e.setAttribute("role", "presentation"), e.textContent = t.str, e.dir = t.dir, o(this, jf) && (e.dataset.fontName = l.fontSubstitutionLoadedName || t.fontName), a !== 0 && (n.angle = a * (180 / Math.PI));
  let x = !1;
  if (t.str.length > 1)
    x = !0;
  else if (t.str !== " " && t.transform[0] !== t.transform[3]) {
    const k = Math.abs(t.transform[0]), T = Math.abs(t.transform[3]);
    k !== T && Math.max(k, T) / Math.min(k, T) > 1.5 && (x = !0);
  }
  if (x && (n.canvasWidth = l.vertical ? t.height : t.width), o(this, Gf).set(e, n), o(this, jr).div = e, o(this, jr).properties = n, E(this, uo, mw).call(this, o(this, jr)), n.hasText && o(this, ni).append(e), n.hasEOL) {
    const k = document.createElement("br");
    k.setAttribute("role", "presentation"), o(this, ni).append(k);
  }
}, mw = function(t) {
  var u;
  const {
    div: e,
    properties: n,
    ctx: s
  } = t, {
    style: a
  } = e;
  let l = "";
  if (o(Le, na) > 1 && (l = `scale(${1 / o(Le, na)})`), n.canvasWidth !== 0 && n.hasText) {
    const {
      fontFamily: h
    } = a, {
      canvasWidth: f,
      fontSize: g
    } = n;
    E(u = Le, Mi, yw).call(u, s, g * o(this, ea), h);
    const {
      width: y
    } = s.measureText(e.textContent);
    y > 0 && (l = `scaleX(${f * o(this, ea) / y}) ${l}`);
  }
  n.angle !== 0 && (l = `rotate(${n.angle}deg) ${l}`), l.length > 0 && (a.transform = l);
}, Mi = new WeakSet(), Sm = function(t = null) {
  let e = o(this, Rl).get(t || (t = ""));
  if (!e) {
    const n = document.createElement("canvas");
    n.className = "hiddenCanvasElement", n.lang = t, document.body.append(n), e = n.getContext("2d", {
      alpha: !1,
      willReadFrequently: !0
    }), o(this, Rl).set(t, e), o(this, qf).set(e, {
      size: 0,
      family: ""
    });
  }
  return e;
}, yw = function(t, e, n) {
  const s = o(this, qf).get(t);
  e === s.size && n === s.family || (t.font = `${e}px ${n}`, s.size = e, s.family = n);
}, x_ = function() {
  if (o(this, na) !== null)
    return;
  const t = document.createElement("div");
  t.style.opacity = 0, t.style.lineHeight = 1, t.style.fontSize = "1px", t.style.position = "absolute", t.textContent = "X", document.body.append(t), w(this, na, t.getBoundingClientRect().height), t.remove();
}, T_ = function(t, e, n) {
  const s = o(this, Wu).get(t);
  if (s)
    return s;
  const a = E(this, Mi, Sm).call(this, n);
  a.canvas.width = a.canvas.height = ZE, E(this, Mi, yw).call(this, a, ZE, t);
  const l = a.measureText(""), u = l.fontBoundingBoxAscent, h = Math.abs(l.fontBoundingBoxDescent);
  a.canvas.width = a.canvas.height = 0;
  let f = 0.8;
  return u ? f = u / (u + h) : (cn.platform.isFirefox && Rt("Enable the `dom.textMetrics.fontBoundingBox.enabled` preference in `about:config` to improve TextLayer rendering."), e.ascent ? f = e.ascent : e.descent && (f = 1 + e.descent)), o(this, Wu).set(t, f), f;
}, b(Le, Mi), b(Le, Wu, /* @__PURE__ */ new Map()), b(Le, Rl, /* @__PURE__ */ new Map()), b(Le, qf, /* @__PURE__ */ new WeakMap()), b(Le, na, null), b(Le, Gu, /* @__PURE__ */ new Set()), Le);
const wP = 100;
function yA(p = {}) {
  typeof p == "string" || p instanceof URL ? p = {
    url: p
  } : (p instanceof ArrayBuffer || ArrayBuffer.isView(p)) && (p = {
    data: p
  });
  const t = new vw(), {
    docId: e
  } = t, n = p.url ? Tk(p.url) : null, s = p.data ? kk(p.data) : null, a = p.httpHeaders || null, l = p.withCredentials === !0, u = p.password ?? null, h = p.range instanceof vA ? p.range : null, f = Number.isInteger(p.rangeChunkSize) && p.rangeChunkSize > 0 ? p.rangeChunkSize : 2 ** 16;
  let g = p.worker instanceof Rh ? p.worker : null;
  const y = p.verbosity, v = typeof p.docBaseUrl == "string" && !Op(p.docBaseUrl) ? p.docBaseUrl : null, S = nm(p.cMapUrl), x = p.cMapPacked !== !1, _ = p.CMapReaderFactory || (Hn ? Fk : BE), k = nm(p.iccUrl), T = nm(p.standardFontDataUrl), R = p.StandardFontDataFactory || (Hn ? Nk : zE), P = nm(p.wasmUrl), L = p.WasmFactory || (Hn ? Ok : UE), D = p.stopAtErrors !== !0, N = Number.isInteger(p.maxImageSize) && p.maxImageSize > -1 ? p.maxImageSize : -1, B = p.isEvalSupported !== !1, $ = typeof p.isOffscreenCanvasSupported == "boolean" ? p.isOffscreenCanvasSupported : !Hn, V = typeof p.isImageDecoderSupported == "boolean" ? p.isImageDecoderSupported : !Hn && (cn.platform.isFirefox || !globalThis.chrome), X = Number.isInteger(p.canvasMaxAreaInBytes) ? p.canvasMaxAreaInBytes : -1, q = typeof p.disableFontFace == "boolean" ? p.disableFontFace : Hn, it = p.fontExtraProperties === !0, lt = p.enableXfa === !0, rt = p.ownerDocument || globalThis.document, vt = p.disableRange === !0, tt = p.disableStream === !0, Q = p.disableAutoFetch === !0, W = p.pdfBug === !0, ot = p.CanvasFactory || (Hn ? Ik : Mk), st = p.FilterFactory || (Hn ? Dk : Lk), I = p.enableHWA === !0, z = p.useWasm !== !1, K = h ? h.length : p.length ?? NaN, ct = typeof p.useSystemFonts == "boolean" ? p.useSystemFonts : !Hn && !q, Et = typeof p.useWorkerFetch == "boolean" ? p.useWorkerFetch : !!(_ === BE && R === zE && L === UE && S && T && P && Ed(S, document.baseURI) && Ed(T, document.baseURI) && Ed(P, document.baseURI)), xt = null;
  lk(y);
  const bt = {
    canvasFactory: new ot({
      ownerDocument: rt,
      enableHWA: I
    }),
    filterFactory: new st({
      docId: e,
      ownerDocument: rt
    }),
    cMapReaderFactory: Et ? null : new _({
      baseUrl: S,
      isCompressed: x
    }),
    standardFontDataFactory: Et ? null : new R({
      baseUrl: T
    }),
    wasmFactory: Et ? null : new L({
      baseUrl: P
    })
  };
  g || (g = Rh.create({
    verbosity: y,
    port: ts.workerPort
  }), t._worker = g);
  const Ot = {
    docId: e,
    apiVersion: "5.4.296",
    data: s,
    password: u,
    disableAutoFetch: Q,
    rangeChunkSize: f,
    length: K,
    docBaseUrl: v,
    enableXfa: lt,
    evaluatorOptions: {
      maxImageSize: N,
      disableFontFace: q,
      ignoreErrors: D,
      isEvalSupported: B,
      isOffscreenCanvasSupported: $,
      isImageDecoderSupported: V,
      canvasMaxAreaInBytes: X,
      fontExtraProperties: it,
      useSystemFonts: ct,
      useWasm: z,
      useWorkerFetch: Et,
      cMapUrl: S,
      iccUrl: k,
      standardFontDataUrl: T,
      wasmUrl: P
    }
  }, Bt = {
    ownerDocument: rt,
    pdfBug: W,
    styleElement: xt,
    loadingParams: {
      disableAutoFetch: Q,
      enableXfa: lt
    }
  };
  return g.promise.then(function() {
    if (t.destroyed)
      throw new Error("Loading aborted");
    if (g.destroyed)
      throw new Error("Worker was destroyed");
    const Dt = g.messageHandler.sendWithPromise("GetDocRequest", Ot, s ? [s.buffer] : null);
    let Qe;
    if (h)
      Qe = new eP(h, {
        disableRange: vt,
        disableStream: tt
      });
    else if (!s) {
      if (!n)
        throw new Error("getDocument - no `url` parameter provided.");
      const Di = Ed(n) ? rP : Hn ? gP : uP;
      Qe = new Di({
        url: n,
        length: K,
        httpHeaders: a,
        withCredentials: l,
        rangeChunkSize: f,
        disableRange: vt,
        disableStream: tt
      });
    }
    return Dt.then((Di) => {
      if (t.destroyed)
        throw new Error("Loading aborted");
      if (g.destroyed)
        throw new Error("Worker was destroyed");
      const Ht = new Pd(e, Di, g.port), Jt = new bP(Ht, t, Qe, Bt, bt, I);
      t._transport = Jt, Ht.send("Ready", null);
    });
  }).catch(t._capability.reject), t;
}
var uy;
const hy = class hy {
  constructor() {
    Z(this, "_capability", Promise.withResolvers());
    Z(this, "_transport", null);
    Z(this, "_worker", null);
    Z(this, "docId", `d${Ge(hy, uy)._++}`);
    Z(this, "destroyed", !1);
    Z(this, "onPassword", null);
    Z(this, "onProgress", null);
  }
  get promise() {
    return this._capability.promise;
  }
  async destroy() {
    var t, e, n, s;
    this.destroyed = !0;
    try {
      (t = this._worker) != null && t.port && (this._worker._pendingDestroy = !0), await ((e = this._transport) == null ? void 0 : e.destroy());
    } catch (a) {
      throw (n = this._worker) != null && n.port && delete this._worker._pendingDestroy, a;
    }
    this._transport = null, (s = this._worker) == null || s.destroy(), this._worker = null;
  }
  async getData() {
    return this._transport.getData();
  }
};
uy = new WeakMap(), b(hy, uy, 0);
let vw = hy;
var Ml, Yf, Kf, Qf, Zf, g1;
let vA = (g1 = class {
  constructor(t, e, n = !1, s = null) {
    b(this, Ml, Promise.withResolvers());
    b(this, Yf, []);
    b(this, Kf, []);
    b(this, Qf, []);
    b(this, Zf, []);
    this.length = t, this.initialData = e, this.progressiveDone = n, this.contentDispositionFilename = s;
  }
  addRangeListener(t) {
    o(this, Zf).push(t);
  }
  addProgressListener(t) {
    o(this, Qf).push(t);
  }
  addProgressiveReadListener(t) {
    o(this, Kf).push(t);
  }
  addProgressiveDoneListener(t) {
    o(this, Yf).push(t);
  }
  onDataRange(t, e) {
    for (const n of o(this, Zf))
      n(t, e);
  }
  onDataProgress(t, e) {
    o(this, Ml).promise.then(() => {
      for (const n of o(this, Qf))
        n(t, e);
    });
  }
  onDataProgressiveRead(t) {
    o(this, Ml).promise.then(() => {
      for (const e of o(this, Kf))
        e(t);
    });
  }
  onDataProgressiveDone() {
    o(this, Ml).promise.then(() => {
      for (const t of o(this, Yf))
        t();
    });
  }
  transportReady() {
    o(this, Ml).resolve();
  }
  requestDataRange(t, e) {
    ce("Abstract method PDFDataRangeTransport.requestDataRange");
  }
  abort() {
  }
}, Ml = new WeakMap(), Yf = new WeakMap(), Kf = new WeakMap(), Qf = new WeakMap(), Zf = new WeakMap(), g1);
class AP {
  constructor(t, e) {
    this._pdfInfo = t, this._transport = e;
  }
  get annotationStorage() {
    return this._transport.annotationStorage;
  }
  get canvasFactory() {
    return this._transport.canvasFactory;
  }
  get filterFactory() {
    return this._transport.filterFactory;
  }
  get numPages() {
    return this._pdfInfo.numPages;
  }
  get fingerprints() {
    return this._pdfInfo.fingerprints;
  }
  get isPureXfa() {
    return It(this, "isPureXfa", !!this._transport._htmlForXfa);
  }
  get allXfaHtml() {
    return this._transport._htmlForXfa;
  }
  getPage(t) {
    return this._transport.getPage(t);
  }
  getPageIndex(t) {
    return this._transport.getPageIndex(t);
  }
  getDestinations() {
    return this._transport.getDestinations();
  }
  getDestination(t) {
    return this._transport.getDestination(t);
  }
  getPageLabels() {
    return this._transport.getPageLabels();
  }
  getPageLayout() {
    return this._transport.getPageLayout();
  }
  getPageMode() {
    return this._transport.getPageMode();
  }
  getViewerPreferences() {
    return this._transport.getViewerPreferences();
  }
  getOpenAction() {
    return this._transport.getOpenAction();
  }
  getAttachments() {
    return this._transport.getAttachments();
  }
  getAnnotationsByType(t, e) {
    return this._transport.getAnnotationsByType(t, e);
  }
  getJSActions() {
    return this._transport.getDocJSActions();
  }
  getOutline() {
    return this._transport.getOutline();
  }
  getOptionalContentConfig({
    intent: t = "display"
  } = {}) {
    const {
      renderingIntent: e
    } = this._transport.getRenderingIntent(t);
    return this._transport.getOptionalContentConfig(e);
  }
  getPermissions() {
    return this._transport.getPermissions();
  }
  getMetadata() {
    return this._transport.getMetadata();
  }
  getMarkInfo() {
    return this._transport.getMarkInfo();
  }
  getData() {
    return this._transport.getData();
  }
  saveDocument() {
    return this._transport.saveDocument();
  }
  getDownloadInfo() {
    return this._transport.downloadInfoCapability.promise;
  }
  cleanup(t = !1) {
    return this._transport.startCleanup(t || this.isPureXfa);
  }
  destroy() {
    return this.loadingTask.destroy();
  }
  cachedPageNumber(t) {
    return this._transport.cachedPageNumber(t);
  }
  get loadingParams() {
    return this._transport.loadingParams;
  }
  get loadingTask() {
    return this._transport.loadingTask;
  }
  getFieldObjects() {
    return this._transport.getFieldObjects();
  }
  hasJSActions() {
    return this._transport.hasJSActions();
  }
  getCalculationOrderIds() {
    return this._transport.getCalculationOrderIds();
  }
}
var $r, Ll, Nd;
class SP {
  constructor(t, e, n, s = !1) {
    b(this, Ll);
    b(this, $r, !1);
    this._pageIndex = t, this._pageInfo = e, this._transport = n, this._stats = s ? new ME() : null, this._pdfBug = s, this.commonObjs = n.commonObjs, this.objs = new E_(), this._intentStates = /* @__PURE__ */ new Map(), this.destroyed = !1, this.recordedBBoxes = null;
  }
  get pageNumber() {
    return this._pageIndex + 1;
  }
  get rotate() {
    return this._pageInfo.rotate;
  }
  get ref() {
    return this._pageInfo.ref;
  }
  get userUnit() {
    return this._pageInfo.userUnit;
  }
  get view() {
    return this._pageInfo.view;
  }
  getViewport({
    scale: t,
    rotation: e = this.rotate,
    offsetX: n = 0,
    offsetY: s = 0,
    dontFlip: a = !1
  } = {}) {
    return new Np({
      viewBox: this.view,
      userUnit: this.userUnit,
      scale: t,
      rotation: e,
      offsetX: n,
      offsetY: s,
      dontFlip: a
    });
  }
  getAnnotations({
    intent: t = "display"
  } = {}) {
    const {
      renderingIntent: e
    } = this._transport.getRenderingIntent(t);
    return this._transport.getAnnotations(this._pageIndex, e);
  }
  getJSActions() {
    return this._transport.getPageJSActions(this._pageIndex);
  }
  get filterFactory() {
    return this._transport.filterFactory;
  }
  get isPureXfa() {
    return It(this, "isPureXfa", !!this._transport._htmlForXfa);
  }
  async getXfa() {
    var t;
    return ((t = this._transport._htmlForXfa) == null ? void 0 : t.children[this._pageIndex]) || null;
  }
  render({
    canvasContext: t,
    canvas: e = t.canvas,
    viewport: n,
    intent: s = "display",
    annotationMode: a = lr.ENABLE,
    transform: l = null,
    background: u = null,
    optionalContentConfigPromise: h = null,
    annotationCanvasMap: f = null,
    pageColors: g = null,
    printAnnotationStorage: y = null,
    isEditing: v = !1,
    recordOperations: S = !1,
    operationsFilter: x = null
  }) {
    var V, X, q;
    (V = this._stats) == null || V.time("Overall");
    const _ = this._transport.getRenderingIntent(s, a, y, v), {
      renderingIntent: k,
      cacheKey: T
    } = _;
    w(this, $r, !1), h || (h = this._transport.getOptionalContentConfig(k));
    let R = this._intentStates.get(T);
    R || (R = /* @__PURE__ */ Object.create(null), this._intentStates.set(T, R)), R.streamReaderCancelTimeout && (clearTimeout(R.streamReaderCancelTimeout), R.streamReaderCancelTimeout = null);
    const P = !!(k & ki.PRINT);
    R.displayReadyCapability || (R.displayReadyCapability = Promise.withResolvers(), R.operatorList = {
      fnArray: [],
      argsArray: [],
      lastChunk: !1,
      separateAnnots: null
    }, (X = this._stats) == null || X.time("Page Request"), this._pumpOperatorList(_));
    const L = !!(this._pdfBug && ((q = globalThis.StepperManager) != null && q.enabled)), D = !this.recordedBBoxes && (S || L), N = (it) => {
      var lt, rt;
      if (R.renderTasks.delete(B), D) {
        const vt = (lt = B.gfx) == null ? void 0 : lt.dependencyTracker.take();
        vt && (B.stepper && B.stepper.setOperatorBBoxes(vt, B.gfx.dependencyTracker.takeDebugMetadata()), S && (this.recordedBBoxes = vt));
      }
      P && w(this, $r, !0), E(this, Ll, Nd).call(this), it ? (B.capability.reject(it), this._abortOperatorList({
        intentState: R,
        reason: it instanceof Error ? it : new Error(it)
      })) : B.capability.resolve(), this._stats && (this._stats.timeEnd("Rendering"), this._stats.timeEnd("Overall"), (rt = globalThis.Stats) != null && rt.enabled && globalThis.Stats.add(this.pageNumber, this._stats));
    }, B = new ww({
      callback: N,
      params: {
        canvas: e,
        canvasContext: t,
        dependencyTracker: D ? new zk(e, R.operatorList.length, L) : null,
        viewport: n,
        transform: l,
        background: u
      },
      objs: this.objs,
      commonObjs: this.commonObjs,
      annotationCanvasMap: f,
      operatorList: R.operatorList,
      pageIndex: this._pageIndex,
      canvasFactory: this._transport.canvasFactory,
      filterFactory: this._transport.filterFactory,
      useRequestAnimationFrame: !P,
      pdfBug: this._pdfBug,
      pageColors: g,
      enableHWA: this._transport.enableHWA,
      operationsFilter: x
    });
    (R.renderTasks || (R.renderTasks = /* @__PURE__ */ new Set())).add(B);
    const $ = B.task;
    return Promise.all([R.displayReadyCapability.promise, h]).then(([it, lt]) => {
      var rt;
      if (this.destroyed) {
        N();
        return;
      }
      if ((rt = this._stats) == null || rt.time("Rendering"), !(lt.renderingIntent & k))
        throw new Error("Must use the same `intent`-argument when calling the `PDFPageProxy.render` and `PDFDocumentProxy.getOptionalContentConfig` methods.");
      B.initializeGraphics({
        transparency: it,
        optionalContentConfig: lt
      }), B.operatorListChanged();
    }).catch(N), $;
  }
  getOperatorList({
    intent: t = "display",
    annotationMode: e = lr.ENABLE,
    printAnnotationStorage: n = null,
    isEditing: s = !1
  } = {}) {
    var f;
    function a() {
      u.operatorList.lastChunk && (u.opListReadCapability.resolve(u.operatorList), u.renderTasks.delete(h));
    }
    const l = this._transport.getRenderingIntent(t, e, n, s, !0);
    let u = this._intentStates.get(l.cacheKey);
    u || (u = /* @__PURE__ */ Object.create(null), this._intentStates.set(l.cacheKey, u));
    let h;
    return u.opListReadCapability || (h = /* @__PURE__ */ Object.create(null), h.operatorListChanged = a, u.opListReadCapability = Promise.withResolvers(), (u.renderTasks || (u.renderTasks = /* @__PURE__ */ new Set())).add(h), u.operatorList = {
      fnArray: [],
      argsArray: [],
      lastChunk: !1,
      separateAnnots: null
    }, (f = this._stats) == null || f.time("Page Request"), this._pumpOperatorList(l)), u.opListReadCapability.promise;
  }
  streamTextContent({
    includeMarkedContent: t = !1,
    disableNormalization: e = !1
  } = {}) {
    return this._transport.messageHandler.sendWithStream("GetTextContent", {
      pageIndex: this._pageIndex,
      includeMarkedContent: t === !0,
      disableNormalization: e === !0
    }, {
      highWaterMark: 100,
      size(s) {
        return s.items.length;
      }
    });
  }
  getTextContent(t = {}) {
    if (this._transport._htmlForXfa)
      return this.getXfa().then((n) => Xd.textContent(n));
    const e = this.streamTextContent(t);
    return new Promise(function(n, s) {
      function a() {
        l.read().then(function({
          value: h,
          done: f
        }) {
          if (f) {
            n(u);
            return;
          }
          u.lang ?? (u.lang = h.lang), Object.assign(u.styles, h.styles), u.items.push(...h.items), a();
        }, s);
      }
      const l = e.getReader(), u = {
        items: [],
        styles: /* @__PURE__ */ Object.create(null),
        lang: null
      };
      a();
    });
  }
  getStructTree() {
    return this._transport.getStructTree(this._pageIndex);
  }
  _destroy() {
    this.destroyed = !0;
    const t = [];
    for (const e of this._intentStates.values())
      if (this._abortOperatorList({
        intentState: e,
        reason: new Error("Page was destroyed."),
        force: !0
      }), !e.opListReadCapability)
        for (const n of e.renderTasks)
          t.push(n.completed), n.cancel();
    return this.objs.clear(), w(this, $r, !1), Promise.all(t);
  }
  cleanup(t = !1) {
    w(this, $r, !0);
    const e = E(this, Ll, Nd).call(this);
    return t && e && this._stats && (this._stats = new ME()), e;
  }
  _startRenderPage(t, e) {
    var s, a;
    const n = this._intentStates.get(e);
    n && ((s = this._stats) == null || s.timeEnd("Page Request"), (a = n.displayReadyCapability) == null || a.resolve(t));
  }
  _renderPageChunk(t, e) {
    for (let n = 0, s = t.length; n < s; n++)
      e.operatorList.fnArray.push(t.fnArray[n]), e.operatorList.argsArray.push(t.argsArray[n]);
    e.operatorList.lastChunk = t.lastChunk, e.operatorList.separateAnnots = t.separateAnnots;
    for (const n of e.renderTasks)
      n.operatorListChanged();
    t.lastChunk && E(this, Ll, Nd).call(this);
  }
  _pumpOperatorList({
    renderingIntent: t,
    cacheKey: e,
    annotationStorageSerializable: n,
    modifiedIds: s
  }) {
    const {
      map: a,
      transfer: l
    } = n, h = this._transport.messageHandler.sendWithStream("GetOperatorList", {
      pageIndex: this._pageIndex,
      intent: t,
      cacheKey: e,
      annotationStorage: a,
      modifiedIds: s
    }, l).getReader(), f = this._intentStates.get(e);
    f.streamReader = h;
    const g = () => {
      h.read().then(({
        value: y,
        done: v
      }) => {
        if (v) {
          f.streamReader = null;
          return;
        }
        this._transport.destroyed || (this._renderPageChunk(y, f), g());
      }, (y) => {
        if (f.streamReader = null, !this._transport.destroyed) {
          if (f.operatorList) {
            f.operatorList.lastChunk = !0;
            for (const v of f.renderTasks)
              v.operatorListChanged();
            E(this, Ll, Nd).call(this);
          }
          if (f.displayReadyCapability)
            f.displayReadyCapability.reject(y);
          else if (f.opListReadCapability)
            f.opListReadCapability.reject(y);
          else
            throw y;
        }
      });
    };
    g();
  }
  _abortOperatorList({
    intentState: t,
    reason: e,
    force: n = !1
  }) {
    if (t.streamReader) {
      if (t.streamReaderCancelTimeout && (clearTimeout(t.streamReaderCancelTimeout), t.streamReaderCancelTimeout = null), !n) {
        if (t.renderTasks.size > 0)
          return;
        if (e instanceof by) {
          let s = wP;
          e.extraDelay > 0 && e.extraDelay < 1e3 && (s += e.extraDelay), t.streamReaderCancelTimeout = setTimeout(() => {
            t.streamReaderCancelTimeout = null, this._abortOperatorList({
              intentState: t,
              reason: e,
              force: !0
            });
          }, s);
          return;
        }
      }
      if (t.streamReader.cancel(new ao(e.message)).catch(() => {
      }), t.streamReader = null, !this._transport.destroyed) {
        for (const [s, a] of this._intentStates)
          if (a === t) {
            this._intentStates.delete(s);
            break;
          }
        this.cleanup();
      }
    }
  }
  get stats() {
    return this._stats;
  }
}
$r = new WeakMap(), Ll = new WeakSet(), Nd = function() {
  if (!o(this, $r) || this.destroyed)
    return !1;
  for (const {
    renderTasks: t,
    operatorList: e
  } of this._intentStates.values())
    if (t.size > 0 || !e.lastChunk)
      return !1;
  return this._intentStates.clear(), this.objs.clear(), w(this, $r, !1), !0;
};
var ia, As, Vr, Dl, dy, Il, Fl, jn, bm, k_, P_, Od, Xu, Em;
const ve = class ve {
  constructor({
    name: t = null,
    port: e = null,
    verbosity: n = ck()
  } = {}) {
    b(this, jn);
    b(this, ia, Promise.withResolvers());
    b(this, As, null);
    b(this, Vr, null);
    b(this, Dl, null);
    if (this.name = t, this.destroyed = !1, this.verbosity = n, e) {
      if (o(ve, Fl).has(e))
        throw new Error("Cannot use more than one PDFWorker per port.");
      o(ve, Fl).set(e, this), E(this, jn, k_).call(this, e);
    } else
      E(this, jn, P_).call(this);
  }
  get promise() {
    return o(this, ia).promise;
  }
  get port() {
    return o(this, Vr);
  }
  get messageHandler() {
    return o(this, As);
  }
  destroy() {
    var t, e;
    this.destroyed = !0, (t = o(this, Dl)) == null || t.terminate(), w(this, Dl, null), o(ve, Fl).delete(o(this, Vr)), w(this, Vr, null), (e = o(this, As)) == null || e.destroy(), w(this, As, null);
  }
  static create(t) {
    const e = o(this, Fl).get(t == null ? void 0 : t.port);
    if (e) {
      if (e._pendingDestroy)
        throw new Error("PDFWorker.create - the worker is being destroyed.\nPlease remember to await `PDFDocumentLoadingTask.destroy()`-calls.");
      return e;
    }
    return new ve(t);
  }
  static get workerSrc() {
    if (ts.workerSrc)
      return ts.workerSrc;
    throw new Error('No "GlobalWorkerOptions.workerSrc" specified.');
  }
  static get _setupFakeWorkerGlobal() {
    return It(this, "_setupFakeWorkerGlobal", (async () => o(this, Xu, Em) ? o(this, Xu, Em) : (await import(
      /*webpackIgnore: true*/
      /*@vite-ignore*/
      this.workerSrc
    )).WorkerMessageHandler)());
  }
};
ia = new WeakMap(), As = new WeakMap(), Vr = new WeakMap(), Dl = new WeakMap(), dy = new WeakMap(), Il = new WeakMap(), Fl = new WeakMap(), jn = new WeakSet(), bm = function() {
  o(this, ia).resolve(), o(this, As).send("configure", {
    verbosity: this.verbosity
  });
}, k_ = function(t) {
  w(this, Vr, t), w(this, As, new Pd("main", "worker", t)), o(this, As).on("ready", () => {
  }), E(this, jn, bm).call(this);
}, P_ = function() {
  if (o(ve, Il) || o(ve, Xu, Em)) {
    E(this, jn, Od).call(this);
    return;
  }
  let {
    workerSrc: t
  } = ve;
  try {
    ve._isSameOrigin(window.location, t) || (t = ve._createCDNWrapper(new URL(t, window.location).href));
    const e = new Worker(t, {
      type: "module"
    }), n = new Pd("main", "worker", e), s = () => {
      a.abort(), n.destroy(), e.terminate(), this.destroyed ? o(this, ia).reject(new Error("Worker was destroyed")) : E(this, jn, Od).call(this);
    }, a = new AbortController();
    e.addEventListener("error", () => {
      o(this, Dl) || s();
    }, {
      signal: a.signal
    }), n.on("test", (u) => {
      if (a.abort(), this.destroyed || !u) {
        s();
        return;
      }
      w(this, As, n), w(this, Vr, e), w(this, Dl, e), E(this, jn, bm).call(this);
    }), n.on("ready", (u) => {
      if (a.abort(), this.destroyed) {
        s();
        return;
      }
      try {
        l();
      } catch {
        E(this, jn, Od).call(this);
      }
    });
    const l = () => {
      const u = new Uint8Array();
      n.send("test", u, [u.buffer]);
    };
    l();
    return;
  } catch {
    Sy("The worker has been disabled.");
  }
  E(this, jn, Od).call(this);
}, Od = function() {
  o(ve, Il) || (Rt("Setting up fake worker."), w(ve, Il, !0)), ve._setupFakeWorkerGlobal.then((t) => {
    if (this.destroyed) {
      o(this, ia).reject(new Error("Worker was destroyed"));
      return;
    }
    const e = new Rk();
    w(this, Vr, e);
    const n = `fake${Ge(ve, dy)._++}`, s = new Pd(n + "_worker", n, e);
    t.setup(s, e), w(this, As, new Pd(n, n + "_worker", e)), E(this, jn, bm).call(this);
  }).catch((t) => {
    o(this, ia).reject(new Error(`Setting up fake worker failed: "${t.message}".`));
  });
}, Xu = new WeakSet(), Em = function() {
  var t;
  try {
    return ((t = globalThis.pdfjsWorker) == null ? void 0 : t.WorkerMessageHandler) || null;
  } catch {
    return null;
  }
}, b(ve, Xu), b(ve, dy, 0), b(ve, Il, !1), b(ve, Fl, /* @__PURE__ */ new WeakMap()), Hn && (w(ve, Il, !0), ts.workerSrc || (ts.workerSrc = "./pdf.worker.mjs")), ve._isSameOrigin = (t, e) => {
  const n = URL.parse(t);
  if (!(n != null && n.origin) || n.origin === "null")
    return !1;
  const s = new URL(e, n);
  return n.origin === s.origin;
}, ve._createCDNWrapper = (t) => {
  const e = `await import("${t}");`;
  return URL.createObjectURL(new Blob([e], {
    type: "text/javascript"
  }));
}, ve.fromPort = (t) => {
  if (vk("`PDFWorker.fromPort` - please use `PDFWorker.create` instead."), !(t != null && t.port))
    throw new Error("PDFWorker.fromPort - invalid method signature.");
  return ve.create(t);
};
let Rh = ve;
var Wr, Zs, qu, Yu, Gr, Nl, Bd;
class bP {
  constructor(t, e, n, s, a, l) {
    b(this, Nl);
    b(this, Wr, /* @__PURE__ */ new Map());
    b(this, Zs, /* @__PURE__ */ new Map());
    b(this, qu, /* @__PURE__ */ new Map());
    b(this, Yu, /* @__PURE__ */ new Map());
    b(this, Gr, null);
    this.messageHandler = t, this.loadingTask = e, this.commonObjs = new E_(), this.fontLoader = new Ck({
      ownerDocument: s.ownerDocument,
      styleElement: s.styleElement
    }), this.loadingParams = s.loadingParams, this._params = s, this.canvasFactory = a.canvasFactory, this.filterFactory = a.filterFactory, this.cMapReaderFactory = a.cMapReaderFactory, this.standardFontDataFactory = a.standardFontDataFactory, this.wasmFactory = a.wasmFactory, this.destroyed = !1, this.destroyCapability = null, this._networkStream = n, this._fullReader = null, this._lastProgress = null, this.downloadInfoCapability = Promise.withResolvers(), this.enableHWA = l, this.setupMessageHandler();
  }
  get annotationStorage() {
    return It(this, "annotationStorage", new pA());
  }
  getRenderingIntent(t, e = lr.ENABLE, n = null, s = !1, a = !1) {
    let l = ki.DISPLAY, u = nw;
    switch (t) {
      case "any":
        l = ki.ANY;
        break;
      case "display":
        break;
      case "print":
        l = ki.PRINT;
        break;
      default:
        Rt(`getRenderingIntent - invalid intent: ${t}`);
    }
    const h = l & ki.PRINT && n instanceof o_ ? n : this.annotationStorage;
    switch (e) {
      case lr.DISABLE:
        l += ki.ANNOTATIONS_DISABLE;
        break;
      case lr.ENABLE:
        break;
      case lr.ENABLE_FORMS:
        l += ki.ANNOTATIONS_FORMS;
        break;
      case lr.ENABLE_STORAGE:
        l += ki.ANNOTATIONS_STORAGE, u = h.serializable;
        break;
      default:
        Rt(`getRenderingIntent - invalid annotationMode: ${e}`);
    }
    s && (l += ki.IS_EDITING), a && (l += ki.OPLIST);
    const {
      ids: f,
      hash: g
    } = h.modifiedIds, y = [l, u.hash, g];
    return {
      renderingIntent: l,
      cacheKey: y.join("_"),
      annotationStorageSerializable: u,
      modifiedIds: f
    };
  }
  destroy() {
    var n;
    if (this.destroyCapability)
      return this.destroyCapability.promise;
    this.destroyed = !0, this.destroyCapability = Promise.withResolvers(), (n = o(this, Gr)) == null || n.reject(new Error("Worker was destroyed during onPassword callback"));
    const t = [];
    for (const s of o(this, Zs).values())
      t.push(s._destroy());
    o(this, Zs).clear(), o(this, qu).clear(), o(this, Yu).clear(), this.hasOwnProperty("annotationStorage") && this.annotationStorage.resetModified();
    const e = this.messageHandler.sendWithPromise("Terminate", null);
    return t.push(e), Promise.all(t).then(() => {
      var s, a;
      this.commonObjs.clear(), this.fontLoader.clear(), o(this, Wr).clear(), this.filterFactory.destroy(), Kd.cleanup(), (s = this._networkStream) == null || s.cancelAllRequests(new ao("Worker was terminated.")), (a = this.messageHandler) == null || a.destroy(), this.messageHandler = null, this.destroyCapability.resolve();
    }, this.destroyCapability.reject), this.destroyCapability.promise;
  }
  setupMessageHandler() {
    const {
      messageHandler: t,
      loadingTask: e
    } = this;
    t.on("GetReader", (n, s) => {
      $t(this._networkStream, "GetReader - no `IPDFStream` instance available."), this._fullReader = this._networkStream.getFullReader(), this._fullReader.onProgress = (a) => {
        this._lastProgress = {
          loaded: a.loaded,
          total: a.total
        };
      }, s.onPull = () => {
        this._fullReader.read().then(function({
          value: a,
          done: l
        }) {
          if (l) {
            s.close();
            return;
          }
          $t(a instanceof ArrayBuffer, "GetReader - expected an ArrayBuffer."), s.enqueue(new Uint8Array(a), 1, [a]);
        }).catch((a) => {
          s.error(a);
        });
      }, s.onCancel = (a) => {
        this._fullReader.cancel(a), s.ready.catch((l) => {
          if (!this.destroyed)
            throw l;
        });
      };
    }), t.on("ReaderHeadersReady", async (n) => {
      var u;
      await this._fullReader.headersReady;
      const {
        isStreamingSupported: s,
        isRangeSupported: a,
        contentLength: l
      } = this._fullReader;
      return (!s || !a) && (this._lastProgress && ((u = e.onProgress) == null || u.call(e, this._lastProgress)), this._fullReader.onProgress = (h) => {
        var f;
        (f = e.onProgress) == null || f.call(e, {
          loaded: h.loaded,
          total: h.total
        });
      }), {
        isStreamingSupported: s,
        isRangeSupported: a,
        contentLength: l
      };
    }), t.on("GetRangeReader", (n, s) => {
      $t(this._networkStream, "GetRangeReader - no `IPDFStream` instance available.");
      const a = this._networkStream.getRangeReader(n.begin, n.end);
      if (!a) {
        s.close();
        return;
      }
      s.onPull = () => {
        a.read().then(function({
          value: l,
          done: u
        }) {
          if (u) {
            s.close();
            return;
          }
          $t(l instanceof ArrayBuffer, "GetRangeReader - expected an ArrayBuffer."), s.enqueue(new Uint8Array(l), 1, [l]);
        }).catch((l) => {
          s.error(l);
        });
      }, s.onCancel = (l) => {
        a.cancel(l), s.ready.catch((u) => {
          if (!this.destroyed)
            throw u;
        });
      };
    }), t.on("GetDoc", ({
      pdfInfo: n
    }) => {
      this._numPages = n.numPages, this._htmlForXfa = n.htmlForXfa, delete n.htmlForXfa, e._capability.resolve(new AP(n, this));
    }), t.on("DocException", (n) => {
      e._capability.reject(Kn(n));
    }), t.on("PasswordRequest", (n) => {
      w(this, Gr, Promise.withResolvers());
      try {
        if (!e.onPassword)
          throw Kn(n);
        const s = (a) => {
          a instanceof Error ? o(this, Gr).reject(a) : o(this, Gr).resolve({
            password: a
          });
        };
        e.onPassword(s, n.code);
      } catch (s) {
        o(this, Gr).reject(s);
      }
      return o(this, Gr).promise;
    }), t.on("DataLoaded", (n) => {
      var s;
      (s = e.onProgress) == null || s.call(e, {
        loaded: n.length,
        total: n.length
      }), this.downloadInfoCapability.resolve(n);
    }), t.on("StartRenderPage", (n) => {
      if (this.destroyed)
        return;
      o(this, Zs).get(n.pageIndex)._startRenderPage(n.transparency, n.cacheKey);
    }), t.on("commonobj", ([n, s, a]) => {
      var l;
      if (this.destroyed || this.commonObjs.has(n))
        return null;
      switch (s) {
        case "Font":
          if ("error" in a) {
            const y = a.error;
            Rt(`Error during font loading: ${y}`), this.commonObjs.resolve(n, y);
            break;
          }
          const u = new fw(a), h = this._params.pdfBug && ((l = globalThis.FontInspector) != null && l.enabled) ? (y, v) => globalThis.FontInspector.fontAdded(y, v) : null, f = new xk(u, h, a.extra, a.charProcOperatorList);
          this.fontLoader.bind(f).catch(() => t.sendWithPromise("FontFallback", {
            id: n
          })).finally(() => {
            !f.fontExtraProperties && f.data && f.clearData(), this.commonObjs.resolve(n, f);
          });
          break;
        case "CopyLocalImage":
          const {
            imageRef: g
          } = a;
          $t(g, "The imageRef must be defined.");
          for (const y of o(this, Zs).values())
            for (const [, v] of y.objs)
              if ((v == null ? void 0 : v.ref) === g)
                return v.dataLen ? (this.commonObjs.resolve(n, structuredClone(v)), v.dataLen) : null;
          break;
        case "FontPath":
        case "Image":
        case "Pattern":
          this.commonObjs.resolve(n, a);
          break;
        default:
          throw new Error(`Got unknown common object type ${s}`);
      }
      return null;
    }), t.on("obj", ([n, s, a, l]) => {
      var h;
      if (this.destroyed)
        return;
      const u = o(this, Zs).get(s);
      if (!u.objs.has(n)) {
        if (u._intentStates.size === 0) {
          (h = l == null ? void 0 : l.bitmap) == null || h.close();
          return;
        }
        switch (a) {
          case "Image":
          case "Pattern":
            u.objs.resolve(n, l);
            break;
          default:
            throw new Error(`Got unknown object type ${a}`);
        }
      }
    }), t.on("DocProgress", (n) => {
      var s;
      this.destroyed || (s = e.onProgress) == null || s.call(e, {
        loaded: n.loaded,
        total: n.total
      });
    }), t.on("FetchBinaryData", async (n) => {
      if (this.destroyed)
        throw new Error("Worker was destroyed.");
      const s = this[n.type];
      if (!s)
        throw new Error(`${n.type} not initialized, see the \`useWorkerFetch\` parameter.`);
      return s.fetch(n);
    });
  }
  getData() {
    return this.messageHandler.sendWithPromise("GetData", null);
  }
  saveDocument() {
    var n;
    this.annotationStorage.size <= 0 && Rt("saveDocument called while `annotationStorage` is empty, please use the getData-method instead.");
    const {
      map: t,
      transfer: e
    } = this.annotationStorage.serializable;
    return this.messageHandler.sendWithPromise("SaveDocument", {
      isPureXfa: !!this._htmlForXfa,
      numPages: this._numPages,
      annotationStorage: t,
      filename: ((n = this._fullReader) == null ? void 0 : n.filename) ?? null
    }, e).finally(() => {
      this.annotationStorage.resetModified();
    });
  }
  getPage(t) {
    if (!Number.isInteger(t) || t <= 0 || t > this._numPages)
      return Promise.reject(new Error("Invalid page request."));
    const e = t - 1, n = o(this, qu).get(e);
    if (n)
      return n;
    const s = this.messageHandler.sendWithPromise("GetPage", {
      pageIndex: e
    }).then((a) => {
      if (this.destroyed)
        throw new Error("Transport destroyed");
      a.refStr && o(this, Yu).set(a.refStr, t);
      const l = new SP(e, a, this, this._params.pdfBug);
      return o(this, Zs).set(e, l), l;
    });
    return o(this, qu).set(e, s), s;
  }
  getPageIndex(t) {
    return iw(t) ? this.messageHandler.sendWithPromise("GetPageIndex", {
      num: t.num,
      gen: t.gen
    }) : Promise.reject(new Error("Invalid pageIndex request."));
  }
  getAnnotations(t, e) {
    return this.messageHandler.sendWithPromise("GetAnnotations", {
      pageIndex: t,
      intent: e
    });
  }
  getFieldObjects() {
    return E(this, Nl, Bd).call(this, "GetFieldObjects");
  }
  hasJSActions() {
    return E(this, Nl, Bd).call(this, "HasJSActions");
  }
  getCalculationOrderIds() {
    return this.messageHandler.sendWithPromise("GetCalculationOrderIds", null);
  }
  getDestinations() {
    return this.messageHandler.sendWithPromise("GetDestinations", null);
  }
  getDestination(t) {
    return typeof t != "string" ? Promise.reject(new Error("Invalid destination request.")) : this.messageHandler.sendWithPromise("GetDestination", {
      id: t
    });
  }
  getPageLabels() {
    return this.messageHandler.sendWithPromise("GetPageLabels", null);
  }
  getPageLayout() {
    return this.messageHandler.sendWithPromise("GetPageLayout", null);
  }
  getPageMode() {
    return this.messageHandler.sendWithPromise("GetPageMode", null);
  }
  getViewerPreferences() {
    return this.messageHandler.sendWithPromise("GetViewerPreferences", null);
  }
  getOpenAction() {
    return this.messageHandler.sendWithPromise("GetOpenAction", null);
  }
  getAttachments() {
    return this.messageHandler.sendWithPromise("GetAttachments", null);
  }
  getAnnotationsByType(t, e) {
    return this.messageHandler.sendWithPromise("GetAnnotationsByType", {
      types: t,
      pageIndexesToSkip: e
    });
  }
  getDocJSActions() {
    return E(this, Nl, Bd).call(this, "GetDocJSActions");
  }
  getPageJSActions(t) {
    return this.messageHandler.sendWithPromise("GetPageJSActions", {
      pageIndex: t
    });
  }
  getStructTree(t) {
    return this.messageHandler.sendWithPromise("GetStructTree", {
      pageIndex: t
    });
  }
  getOutline() {
    return this.messageHandler.sendWithPromise("GetOutline", null);
  }
  getOptionalContentConfig(t) {
    return E(this, Nl, Bd).call(this, "GetOptionalContentConfig").then((e) => new tP(e, t));
  }
  getPermissions() {
    return this.messageHandler.sendWithPromise("GetPermissions", null);
  }
  getMetadata() {
    const t = "GetMetadata", e = o(this, Wr).get(t);
    if (e)
      return e;
    const n = this.messageHandler.sendWithPromise(t, null).then((s) => {
      var a, l;
      return {
        info: s[0],
        metadata: s[1] ? new Zk(s[1]) : null,
        contentDispositionFilename: ((a = this._fullReader) == null ? void 0 : a.filename) ?? null,
        contentLength: ((l = this._fullReader) == null ? void 0 : l.contentLength) ?? null
      };
    });
    return o(this, Wr).set(t, n), n;
  }
  getMarkInfo() {
    return this.messageHandler.sendWithPromise("GetMarkInfo", null);
  }
  async startCleanup(t = !1) {
    if (!this.destroyed) {
      await this.messageHandler.sendWithPromise("Cleanup", null);
      for (const e of o(this, Zs).values())
        if (!e.cleanup())
          throw new Error(`startCleanup: Page ${e.pageNumber} is currently rendering.`);
      this.commonObjs.clear(), t || this.fontLoader.clear(), o(this, Wr).clear(), this.filterFactory.destroy(!0), Kd.cleanup();
    }
  }
  cachedPageNumber(t) {
    if (!iw(t))
      return null;
    const e = t.gen === 0 ? `${t.num}R` : `${t.num}R${t.gen}`;
    return o(this, Yu).get(e) ?? null;
  }
}
Wr = new WeakMap(), Zs = new WeakMap(), qu = new WeakMap(), Yu = new WeakMap(), Gr = new WeakMap(), Nl = new WeakSet(), Bd = function(t, e = null) {
  const n = o(this, Wr).get(t);
  if (n)
    return n;
  const s = this.messageHandler.sendWithPromise(t, e);
  return o(this, Wr).set(t, s), s;
};
var sa;
class EP {
  constructor(t) {
    b(this, sa, null);
    Z(this, "onContinue", null);
    Z(this, "onError", null);
    w(this, sa, t);
  }
  get promise() {
    return o(this, sa).capability.promise;
  }
  cancel(t = 0) {
    o(this, sa).cancel(null, t);
  }
  get separateAnnots() {
    const {
      separateAnnots: t
    } = o(this, sa).operatorList;
    if (!t)
      return !1;
    const {
      annotationCanvasMap: e
    } = o(this, sa);
    return t.form || t.canvas && (e == null ? void 0 : e.size) > 0;
  }
}
sa = new WeakMap();
var ra, Ol;
const Qa = class Qa {
  constructor({
    callback: t,
    params: e,
    objs: n,
    commonObjs: s,
    annotationCanvasMap: a,
    operatorList: l,
    pageIndex: u,
    canvasFactory: h,
    filterFactory: f,
    useRequestAnimationFrame: g = !1,
    pdfBug: y = !1,
    pageColors: v = null,
    enableHWA: S = !1,
    operationsFilter: x = null
  }) {
    b(this, ra, null);
    this.callback = t, this.params = e, this.objs = n, this.commonObjs = s, this.annotationCanvasMap = a, this.operatorListIdx = null, this.operatorList = l, this._pageIndex = u, this.canvasFactory = h, this.filterFactory = f, this._pdfBug = y, this.pageColors = v, this.running = !1, this.graphicsReadyCallback = null, this.graphicsReady = !1, this._useRequestAnimationFrame = g === !0 && typeof window < "u", this.cancelled = !1, this.capability = Promise.withResolvers(), this.task = new EP(this), this._cancelBound = this.cancel.bind(this), this._continueBound = this._continue.bind(this), this._scheduleNextBound = this._scheduleNext.bind(this), this._nextBound = this._next.bind(this), this._canvas = e.canvas, this._canvasContext = e.canvas ? null : e.canvasContext, this._enableHWA = S, this._dependencyTracker = e.dependencyTracker, this._operationsFilter = x;
  }
  get completed() {
    return this.capability.promise.catch(function() {
    });
  }
  initializeGraphics({
    transparency: t = !1,
    optionalContentConfig: e
  }) {
    var h, f;
    if (this.cancelled)
      return;
    if (this._canvas) {
      if (o(Qa, Ol).has(this._canvas))
        throw new Error("Cannot use the same canvas during multiple render() operations. Use different canvas or ensure previous operations were cancelled or completed.");
      o(Qa, Ol).add(this._canvas);
    }
    this._pdfBug && ((h = globalThis.StepperManager) != null && h.enabled) && (this.stepper = globalThis.StepperManager.create(this._pageIndex), this.stepper.init(this.operatorList), this.stepper.nextBreakPoint = this.stepper.getNextBreakPoint());
    const {
      viewport: n,
      transform: s,
      background: a,
      dependencyTracker: l
    } = this.params, u = this._canvasContext || this._canvas.getContext("2d", {
      alpha: !1,
      willReadFrequently: !this._enableHWA
    });
    this.gfx = new tu(u, this.commonObjs, this.objs, this.canvasFactory, this.filterFactory, {
      optionalContentConfig: e
    }, this.annotationCanvasMap, this.pageColors, l), this.gfx.beginDrawing({
      transform: s,
      viewport: n,
      transparency: t,
      background: a
    }), this.operatorListIdx = 0, this.graphicsReady = !0, (f = this.graphicsReadyCallback) == null || f.call(this);
  }
  cancel(t = null, e = 0) {
    var n, s, a;
    this.running = !1, this.cancelled = !0, (n = this.gfx) == null || n.endDrawing(), o(this, ra) && (window.cancelAnimationFrame(o(this, ra)), w(this, ra, null)), o(Qa, Ol).delete(this._canvas), t || (t = new by(`Rendering cancelled, page ${this._pageIndex + 1}`, e)), this.callback(t), (a = (s = this.task).onError) == null || a.call(s, t);
  }
  operatorListChanged() {
    var t, e;
    if (!this.graphicsReady) {
      this.graphicsReadyCallback || (this.graphicsReadyCallback = this._continueBound);
      return;
    }
    (t = this.gfx.dependencyTracker) == null || t.growOperationsCount(this.operatorList.fnArray.length), (e = this.stepper) == null || e.updateOperatorList(this.operatorList), !this.running && this._continue();
  }
  _continue() {
    this.running = !0, !this.cancelled && (this.task.onContinue ? this.task.onContinue(this._scheduleNextBound) : this._scheduleNext());
  }
  _scheduleNext() {
    this._useRequestAnimationFrame ? w(this, ra, window.requestAnimationFrame(() => {
      w(this, ra, null), this._nextBound().catch(this._cancelBound);
    })) : Promise.resolve().then(this._nextBound).catch(this._cancelBound);
  }
  async _next() {
    this.cancelled || (this.operatorListIdx = this.gfx.executeOperatorList(this.operatorList, this.operatorListIdx, this._continueBound, this.stepper, this._operationsFilter), this.operatorListIdx === this.operatorList.argsArray.length && (this.running = !1, this.operatorList.lastChunk && (this.gfx.endDrawing(), o(Qa, Ol).delete(this._canvas), this.callback())));
  }
};
ra = new WeakMap(), Ol = new WeakMap(), b(Qa, Ol, /* @__PURE__ */ new WeakSet());
let ww = Qa;
const R_ = "5.4.296", M_ = "f56dc8601";
var mi, Bl, Ku, Ye, Jf, Qu, Xr, tp, oa, Ss, ep, se, Aw, Sw, bw, Xa, L_, Lo;
const Qn = class Qn {
  constructor({
    editor: t = null,
    uiManager: e = null
  }) {
    b(this, se);
    b(this, mi, null);
    b(this, Bl, null);
    b(this, Ku);
    b(this, Ye, null);
    b(this, Jf, !1);
    b(this, Qu, !1);
    b(this, Xr, null);
    b(this, tp);
    b(this, oa, null);
    b(this, Ss, null);
    var n, s;
    t ? (w(this, Qu, !1), w(this, Xr, t)) : w(this, Qu, !0), w(this, Ss, (t == null ? void 0 : t._uiManager) || e), w(this, tp, o(this, Ss)._eventBus), w(this, Ku, ((n = t == null ? void 0 : t.color) == null ? void 0 : n.toUpperCase()) || ((s = o(this, Ss)) == null ? void 0 : s.highlightColors.values().next().value) || "#FFFF98"), o(Qn, ep) || w(Qn, ep, Object.freeze({
      blue: "pdfjs-editor-colorpicker-blue",
      green: "pdfjs-editor-colorpicker-green",
      pink: "pdfjs-editor-colorpicker-pink",
      red: "pdfjs-editor-colorpicker-red",
      yellow: "pdfjs-editor-colorpicker-yellow"
    }));
  }
  static get _keyboardManager() {
    return It(this, "_keyboardManager", new Bp([[["Escape", "mac+Escape"], Qn.prototype._hideDropdownFromKeyboard], [[" ", "mac+ "], Qn.prototype._colorSelectFromKeyboard], [["ArrowDown", "ArrowRight", "mac+ArrowDown", "mac+ArrowRight"], Qn.prototype._moveToNext], [["ArrowUp", "ArrowLeft", "mac+ArrowUp", "mac+ArrowLeft"], Qn.prototype._moveToPrevious], [["Home", "mac+Home"], Qn.prototype._moveToBeginning], [["End", "mac+End"], Qn.prototype._moveToEnd]]));
  }
  renderButton() {
    const t = w(this, mi, document.createElement("button"));
    t.className = "colorPicker", t.tabIndex = "0", t.setAttribute("data-l10n-id", "pdfjs-editor-colorpicker-button"), t.ariaHasPopup = "true", o(this, Xr) && (t.ariaControls = `${o(this, Xr).id}_colorpicker_dropdown`);
    const e = o(this, Ss)._signal;
    t.addEventListener("click", E(this, se, Xa).bind(this), {
      signal: e
    }), t.addEventListener("keydown", E(this, se, bw).bind(this), {
      signal: e
    });
    const n = w(this, Bl, document.createElement("span"));
    return n.className = "swatch", n.ariaHidden = "true", n.style.backgroundColor = o(this, Ku), t.append(n), t;
  }
  renderMainDropdown() {
    const t = w(this, Ye, E(this, se, Aw).call(this));
    return t.ariaOrientation = "horizontal", t.ariaLabelledBy = "highlightColorPickerLabel", t;
  }
  _colorSelectFromKeyboard(t) {
    if (t.target === o(this, mi)) {
      E(this, se, Xa).call(this, t);
      return;
    }
    const e = t.target.getAttribute("data-color");
    e && E(this, se, Sw).call(this, e, t);
  }
  _moveToNext(t) {
    var e, n;
    if (!o(this, se, Lo)) {
      E(this, se, Xa).call(this, t);
      return;
    }
    if (t.target === o(this, mi)) {
      (e = o(this, Ye).firstChild) == null || e.focus();
      return;
    }
    (n = t.target.nextSibling) == null || n.focus();
  }
  _moveToPrevious(t) {
    var e, n;
    if (t.target === ((e = o(this, Ye)) == null ? void 0 : e.firstChild) || t.target === o(this, mi)) {
      o(this, se, Lo) && this._hideDropdownFromKeyboard();
      return;
    }
    o(this, se, Lo) || E(this, se, Xa).call(this, t), (n = t.target.previousSibling) == null || n.focus();
  }
  _moveToBeginning(t) {
    var e;
    if (!o(this, se, Lo)) {
      E(this, se, Xa).call(this, t);
      return;
    }
    (e = o(this, Ye).firstChild) == null || e.focus();
  }
  _moveToEnd(t) {
    var e;
    if (!o(this, se, Lo)) {
      E(this, se, Xa).call(this, t);
      return;
    }
    (e = o(this, Ye).lastChild) == null || e.focus();
  }
  hideDropdown() {
    var t, e;
    (t = o(this, Ye)) == null || t.classList.add("hidden"), o(this, mi).ariaExpanded = "false", (e = o(this, oa)) == null || e.abort(), w(this, oa, null);
  }
  _hideDropdownFromKeyboard() {
    var t;
    if (!o(this, Qu)) {
      if (!o(this, se, Lo)) {
        (t = o(this, Xr)) == null || t.unselect();
        return;
      }
      this.hideDropdown(), o(this, mi).focus({
        preventScroll: !0,
        focusVisible: o(this, Jf)
      });
    }
  }
  updateColor(t) {
    if (o(this, Bl) && (o(this, Bl).style.backgroundColor = t), !o(this, Ye))
      return;
    const e = o(this, Ss).highlightColors.values();
    for (const n of o(this, Ye).children)
      n.ariaSelected = e.next().value === t.toUpperCase();
  }
  destroy() {
    var t, e;
    (t = o(this, mi)) == null || t.remove(), w(this, mi, null), w(this, Bl, null), (e = o(this, Ye)) == null || e.remove(), w(this, Ye, null);
  }
};
mi = new WeakMap(), Bl = new WeakMap(), Ku = new WeakMap(), Ye = new WeakMap(), Jf = new WeakMap(), Qu = new WeakMap(), Xr = new WeakMap(), tp = new WeakMap(), oa = new WeakMap(), Ss = new WeakMap(), ep = new WeakMap(), se = new WeakSet(), Aw = function() {
  const t = document.createElement("div"), e = o(this, Ss)._signal;
  t.addEventListener("contextmenu", Li, {
    signal: e
  }), t.className = "dropdown", t.role = "listbox", t.ariaMultiSelectable = "false", t.ariaOrientation = "vertical", t.setAttribute("data-l10n-id", "pdfjs-editor-colorpicker-dropdown"), o(this, Xr) && (t.id = `${o(this, Xr).id}_colorpicker_dropdown`);
  for (const [n, s] of o(this, Ss).highlightColors) {
    const a = document.createElement("button");
    a.tabIndex = "0", a.role = "option", a.setAttribute("data-color", s), a.title = n, a.setAttribute("data-l10n-id", o(Qn, ep)[n]);
    const l = document.createElement("span");
    a.append(l), l.className = "swatch", l.style.backgroundColor = s, a.ariaSelected = s === o(this, Ku), a.addEventListener("click", E(this, se, Sw).bind(this, s), {
      signal: e
    }), t.append(a);
  }
  return t.addEventListener("keydown", E(this, se, bw).bind(this), {
    signal: e
  }), t;
}, Sw = function(t, e) {
  e.stopPropagation(), o(this, tp).dispatch("switchannotationeditorparams", {
    source: this,
    type: jt.HIGHLIGHT_COLOR,
    value: t
  }), this.updateColor(t);
}, bw = function(t) {
  Qn._keyboardManager.exec(this, t);
}, Xa = function(t) {
  if (o(this, se, Lo)) {
    this.hideDropdown();
    return;
  }
  if (w(this, Jf, t.detail === 0), o(this, oa) || (w(this, oa, new AbortController()), window.addEventListener("pointerdown", E(this, se, L_).bind(this), {
    signal: o(this, Ss).combinedSignal(o(this, oa))
  })), o(this, mi).ariaExpanded = "true", o(this, Ye)) {
    o(this, Ye).classList.remove("hidden");
    return;
  }
  const e = w(this, Ye, E(this, se, Aw).call(this));
  o(this, mi).append(e);
}, L_ = function(t) {
  var e;
  (e = o(this, Ye)) != null && e.contains(t.target) || this.hideDropdown();
}, Lo = function() {
  return o(this, Ye) && !o(this, Ye).classList.contains("hidden");
}, b(Qn, ep, null);
let Qd = Qn;
var Js, np, Zu, ip;
const Za = class Za {
  constructor(t) {
    b(this, Js, null);
    b(this, np, null);
    b(this, Zu, null);
    w(this, np, t), w(this, Zu, t._uiManager), o(Za, ip) || w(Za, ip, Object.freeze({
      freetext: "pdfjs-editor-color-picker-free-text-input",
      ink: "pdfjs-editor-color-picker-ink-input"
    }));
  }
  renderButton() {
    if (o(this, Js))
      return o(this, Js);
    const {
      editorType: t,
      colorType: e,
      colorValue: n
    } = o(this, np), s = w(this, Js, document.createElement("input"));
    return s.type = "color", s.value = n || "#000000", s.className = "basicColorPicker", s.tabIndex = 0, s.setAttribute("data-l10n-id", o(Za, ip)[t]), s.addEventListener("input", () => {
      o(this, Zu).updateParams(e, s.value);
    }, {
      signal: o(this, Zu)._signal
    }), s;
  }
  update(t) {
    o(this, Js) && (o(this, Js).value = t);
  }
  destroy() {
    var t;
    (t = o(this, Js)) == null || t.remove(), w(this, Js, null);
  }
  hideDropdown() {
  }
};
Js = new WeakMap(), np = new WeakMap(), Zu = new WeakMap(), ip = new WeakMap(), b(Za, ip, null);
let Ym = Za;
function JE(p) {
  return Math.floor(Math.max(0, Math.min(1, p)) * 255).toString(16).padStart(2, "0");
}
function Sd(p) {
  return Math.max(0, Math.min(255, 255 * p));
}
class t1 {
  static CMYK_G([t, e, n, s]) {
    return ["G", 1 - Math.min(1, 0.3 * t + 0.59 * n + 0.11 * e + s)];
  }
  static G_CMYK([t]) {
    return ["CMYK", 0, 0, 0, 1 - t];
  }
  static G_RGB([t]) {
    return ["RGB", t, t, t];
  }
  static G_rgb([t]) {
    return t = Sd(t), [t, t, t];
  }
  static G_HTML([t]) {
    const e = JE(t);
    return `#${e}${e}${e}`;
  }
  static RGB_G([t, e, n]) {
    return ["G", 0.3 * t + 0.59 * e + 0.11 * n];
  }
  static RGB_rgb(t) {
    return t.map(Sd);
  }
  static RGB_HTML(t) {
    return `#${t.map(JE).join("")}`;
  }
  static T_HTML() {
    return "#00000000";
  }
  static T_rgb() {
    return [null];
  }
  static CMYK_RGB([t, e, n, s]) {
    return ["RGB", 1 - Math.min(1, t + s), 1 - Math.min(1, n + s), 1 - Math.min(1, e + s)];
  }
  static CMYK_rgb([t, e, n, s]) {
    return [Sd(1 - Math.min(1, t + s)), Sd(1 - Math.min(1, n + s)), Sd(1 - Math.min(1, e + s))];
  }
  static CMYK_HTML(t) {
    const e = this.CMYK_RGB(t).slice(1);
    return this.RGB_HTML(e);
  }
  static RGB_CMYK([t, e, n]) {
    const s = 1 - t, a = 1 - e, l = 1 - n, u = Math.min(s, a, l);
    return ["CMYK", s, a, l, u];
  }
}
class _P {
  create(t, e, n = !1) {
    if (t <= 0 || e <= 0)
      throw new Error("Invalid SVG dimensions");
    const s = this._createSVG("svg:svg");
    return s.setAttribute("version", "1.1"), n || (s.setAttribute("width", `${t}px`), s.setAttribute("height", `${e}px`)), s.setAttribute("preserveAspectRatio", "none"), s.setAttribute("viewBox", `0 0 ${t} ${e}`), s;
  }
  createElement(t) {
    if (typeof t != "string")
      throw new Error("Invalid SVG element type");
    return this._createSVG(t);
  }
  _createSVG(t) {
    ce("Abstract method `_createSVG` called.");
  }
}
class Zd extends _P {
  _createSVG(t) {
    return document.createElementNS(Ar, t);
  }
}
const CP = 9, hc = /* @__PURE__ */ new WeakSet(), xP = (/* @__PURE__ */ new Date()).getTimezoneOffset() * 60 * 1e3;
class e1 {
  static create(t) {
    switch (t.data.annotationType) {
      case Oe.LINK:
        return new wA(t);
      case Oe.TEXT:
        return new kP(t);
      case Oe.WIDGET:
        switch (t.data.fieldType) {
          case "Tx":
            return new PP(t);
          case "Btn":
            return t.data.radioButton ? new F_(t) : t.data.checkBox ? new MP(t) : new LP(t);
          case "Ch":
            return new DP(t);
          case "Sig":
            return new RP(t);
        }
        return new pc(t);
      case Oe.POPUP:
        return new _w(t);
      case Oe.FREETEXT:
        return new U_(t);
      case Oe.LINE:
        return new FP(t);
      case Oe.SQUARE:
        return new NP(t);
      case Oe.CIRCLE:
        return new OP(t);
      case Oe.POLYLINE:
        return new H_(t);
      case Oe.CARET:
        return new zP(t);
      case Oe.INK:
        return new AA(t);
      case Oe.POLYGON:
        return new BP(t);
      case Oe.HIGHLIGHT:
        return new j_(t);
      case Oe.UNDERLINE:
        return new UP(t);
      case Oe.SQUIGGLY:
        return new HP(t);
      case Oe.STRIKEOUT:
        return new jP(t);
      case Oe.STAMP:
        return new $_(t);
      case Oe.FILEATTACHMENT:
        return new $P(t);
      default:
        return new xe(t);
    }
  }
}
var zl, Ju, Qi, sp, Ew;
const kA = class kA {
  constructor(t, {
    isRenderable: e = !1,
    ignoreBorder: n = !1,
    createQuadrilaterals: s = !1
  } = {}) {
    b(this, sp);
    b(this, zl, null);
    b(this, Ju, !1);
    b(this, Qi, null);
    this.isRenderable = e, this.data = t.data, this.layer = t.layer, this.linkService = t.linkService, this.downloadManager = t.downloadManager, this.imageResourcesPath = t.imageResourcesPath, this.renderForms = t.renderForms, this.svgFactory = t.svgFactory, this.annotationStorage = t.annotationStorage, this.enableComment = t.enableComment, this.enableScripting = t.enableScripting, this.hasJSActions = t.hasJSActions, this._fieldObjects = t.fieldObjects, this.parent = t.parent, e && (this.container = this._createContainer(n)), s && this._createQuadrilaterals();
  }
  static _hasPopupData({
    contentsObj: t,
    richText: e
  }) {
    return !!(t != null && t.str || e != null && e.str);
  }
  get _isEditable() {
    return this.data.isEditable;
  }
  get hasPopupData() {
    return kA._hasPopupData(this.data) || this.enableComment && !!this.commentText;
  }
  get commentData() {
    var n;
    const {
      data: t
    } = this, e = (n = this.annotationStorage) == null ? void 0 : n.getEditor(t.id);
    return e ? e.getData() : t;
  }
  get hasCommentButton() {
    return this.enableComment && this.hasPopupElement;
  }
  get commentButtonPosition() {
    var u;
    const t = (u = this.annotationStorage) == null ? void 0 : u.getEditor(this.data.id);
    if (t)
      return t.commentButtonPositionInPage;
    const {
      quadPoints: e,
      inkLists: n,
      rect: s
    } = this.data;
    let a = -1 / 0, l = -1 / 0;
    if ((e == null ? void 0 : e.length) >= 8) {
      for (let h = 0; h < e.length; h += 8)
        e[h + 1] > l ? (l = e[h + 1], a = e[h + 2]) : e[h + 1] === l && (a = Math.max(a, e[h + 2]));
      return [a, l];
    }
    if ((n == null ? void 0 : n.length) >= 1) {
      for (const h of n)
        for (let f = 0, g = h.length; f < g; f += 2)
          h[f + 1] > l ? (l = h[f + 1], a = h[f]) : h[f + 1] === l && (a = Math.max(a, h[f]));
      if (a !== 1 / 0)
        return [a, l];
    }
    return s ? [s[2], s[3]] : null;
  }
  _normalizePoint(t) {
    const {
      page: {
        view: e
      },
      viewport: {
        rawDims: {
          pageWidth: n,
          pageHeight: s,
          pageX: a,
          pageY: l
        }
      }
    } = this.parent;
    return t[1] = e[3] - t[1] + e[1], t[0] = 100 * (t[0] - a) / n, t[1] = 100 * (t[1] - l) / s, t;
  }
  get commentText() {
    var e, n, s;
    const {
      data: t
    } = this;
    return ((n = (e = this.annotationStorage.getRawValue(`${Wd}${t.id}`)) == null ? void 0 : e.popup) == null ? void 0 : n.contents) || ((s = t.contentsObj) == null ? void 0 : s.str) || "";
  }
  set commentText(t) {
    const {
      data: e
    } = this, n = {
      deleted: !t,
      contents: t || ""
    };
    this.annotationStorage.updateEditor(e.id, {
      popup: n
    }) || this.annotationStorage.setValue(`${Wd}${e.id}`, {
      id: e.id,
      annotationType: e.annotationType,
      pageIndex: this.parent.page._pageIndex,
      popup: n,
      popupRef: e.popupRef,
      modificationDate: /* @__PURE__ */ new Date()
    }), t || this.removePopup();
  }
  removePopup() {
    var t, e;
    (e = ((t = o(this, Qi)) == null ? void 0 : t.popup) || this.popup) == null || e.remove(), w(this, Qi, this.popup = null);
  }
  updateEdited(t) {
    var a;
    if (!this.container)
      return;
    t.rect && (o(this, zl) || w(this, zl, {
      rect: this.data.rect.slice(0)
    }));
    const {
      rect: e,
      popup: n
    } = t;
    e && E(this, sp, Ew).call(this, e);
    let s = ((a = o(this, Qi)) == null ? void 0 : a.popup) || this.popup;
    !s && (n != null && n.text) && (this._createPopup(n), s = o(this, Qi).popup), s && (s.updateEdited(t), n != null && n.deleted && (s.remove(), w(this, Qi, null), this.popup = null));
  }
  resetEdited() {
    var t;
    o(this, zl) && (E(this, sp, Ew).call(this, o(this, zl).rect), (t = o(this, Qi)) == null || t.popup.resetEdited(), w(this, zl, null));
  }
  _createContainer(t) {
    const {
      data: e,
      parent: {
        page: n,
        viewport: s
      }
    } = this, a = document.createElement("section");
    a.setAttribute("data-annotation-id", e.id), !(this instanceof pc) && !(this instanceof wA) && (a.tabIndex = 0);
    const {
      style: l
    } = a;
    if (l.zIndex = this.parent.zIndex, this.parent.zIndex += 2, e.alternativeText && (a.title = e.alternativeText), e.noRotate && a.classList.add("norotate"), !e.rect || this instanceof _w) {
      const {
        rotation: _
      } = e;
      return !e.hasOwnCanvas && _ !== 0 && this.setRotation(_, a), a;
    }
    const {
      width: u,
      height: h
    } = this;
    if (!t && e.borderStyle.width > 0) {
      l.borderWidth = `${e.borderStyle.width}px`;
      const _ = e.borderStyle.horizontalCornerRadius, k = e.borderStyle.verticalCornerRadius;
      if (_ > 0 || k > 0) {
        const R = `calc(${_}px * var(--total-scale-factor)) / calc(${k}px * var(--total-scale-factor))`;
        l.borderRadius = R;
      } else if (this instanceof F_) {
        const R = `calc(${u}px * var(--total-scale-factor)) / calc(${h}px * var(--total-scale-factor))`;
        l.borderRadius = R;
      }
      switch (e.borderStyle.style) {
        case Hc.SOLID:
          l.borderStyle = "solid";
          break;
        case Hc.DASHED:
          l.borderStyle = "dashed";
          break;
        case Hc.BEVELED:
          Rt("Unimplemented border style: beveled");
          break;
        case Hc.INSET:
          Rt("Unimplemented border style: inset");
          break;
        case Hc.UNDERLINE:
          l.borderBottomStyle = "solid";
          break;
      }
      const T = e.borderColor || null;
      T ? (w(this, Ju, !0), l.borderColor = ft.makeHexColor(T[0] | 0, T[1] | 0, T[2] | 0)) : l.borderWidth = 0;
    }
    const f = ft.normalizeRect([e.rect[0], n.view[3] - e.rect[1] + n.view[1], e.rect[2], n.view[3] - e.rect[3] + n.view[1]]), {
      pageWidth: g,
      pageHeight: y,
      pageX: v,
      pageY: S
    } = s.rawDims;
    l.left = `${100 * (f[0] - v) / g}%`, l.top = `${100 * (f[1] - S) / y}%`;
    const {
      rotation: x
    } = e;
    return e.hasOwnCanvas || x === 0 ? (l.width = `${100 * u / g}%`, l.height = `${100 * h / y}%`) : this.setRotation(x, a), a;
  }
  setRotation(t, e = this.container) {
    if (!this.data.rect)
      return;
    const {
      pageWidth: n,
      pageHeight: s
    } = this.parent.viewport.rawDims;
    let {
      width: a,
      height: l
    } = this;
    t % 180 !== 0 && ([a, l] = [l, a]), e.style.width = `${100 * a / n}%`, e.style.height = `${100 * l / s}%`, e.setAttribute("data-main-rotation", (360 - t) % 360);
  }
  get _commonActions() {
    const t = (e, n, s) => {
      const a = s.detail[e], l = a[0], u = a.slice(1);
      s.target.style[n] = t1[`${l}_HTML`](u), this.annotationStorage.setValue(this.data.id, {
        [n]: t1[`${l}_rgb`](u)
      });
    };
    return It(this, "_commonActions", {
      display: (e) => {
        const {
          display: n
        } = e.detail, s = n % 2 === 1;
        this.container.style.visibility = s ? "hidden" : "visible", this.annotationStorage.setValue(this.data.id, {
          noView: s,
          noPrint: n === 1 || n === 2
        });
      },
      print: (e) => {
        this.annotationStorage.setValue(this.data.id, {
          noPrint: !e.detail.print
        });
      },
      hidden: (e) => {
        const {
          hidden: n
        } = e.detail;
        this.container.style.visibility = n ? "hidden" : "visible", this.annotationStorage.setValue(this.data.id, {
          noPrint: n,
          noView: n
        });
      },
      focus: (e) => {
        setTimeout(() => e.target.focus({
          preventScroll: !1
        }), 0);
      },
      userName: (e) => {
        e.target.title = e.detail.userName;
      },
      readonly: (e) => {
        e.target.disabled = e.detail.readonly;
      },
      required: (e) => {
        this._setRequired(e.target, e.detail.required);
      },
      bgColor: (e) => {
        t("bgColor", "backgroundColor", e);
      },
      fillColor: (e) => {
        t("fillColor", "backgroundColor", e);
      },
      fgColor: (e) => {
        t("fgColor", "color", e);
      },
      textColor: (e) => {
        t("textColor", "color", e);
      },
      borderColor: (e) => {
        t("borderColor", "borderColor", e);
      },
      strokeColor: (e) => {
        t("strokeColor", "borderColor", e);
      },
      rotation: (e) => {
        const n = e.detail.rotation;
        this.setRotation(n), this.annotationStorage.setValue(this.data.id, {
          rotation: n
        });
      }
    });
  }
  _dispatchEventFromSandbox(t, e) {
    const n = this._commonActions;
    for (const s of Object.keys(e.detail)) {
      const a = t[s] || n[s];
      a == null || a(e);
    }
  }
  _setDefaultPropertiesFromJS(t) {
    if (!this.enableScripting)
      return;
    const e = this.annotationStorage.getRawValue(this.data.id);
    if (!e)
      return;
    const n = this._commonActions;
    for (const [s, a] of Object.entries(e)) {
      const l = n[s];
      if (l) {
        const u = {
          detail: {
            [s]: a
          },
          target: t
        };
        l(u), delete e[s];
      }
    }
  }
  _createQuadrilaterals() {
    if (!this.container)
      return;
    const {
      quadPoints: t
    } = this.data;
    if (!t)
      return;
    const [e, n, s, a] = this.data.rect.map((_) => Math.fround(_));
    if (t.length === 8) {
      const [_, k, T, R] = t.subarray(2, 6);
      if (s === _ && a === k && e === T && n === R)
        return;
    }
    const {
      style: l
    } = this.container;
    let u;
    if (o(this, Ju)) {
      const {
        borderColor: _,
        borderWidth: k
      } = l;
      l.borderWidth = 0, u = ["url('data:image/svg+xml;utf8,", '<svg xmlns="http://www.w3.org/2000/svg"', ' preserveAspectRatio="none" viewBox="0 0 1 1">', `<g fill="transparent" stroke="${_}" stroke-width="${k}">`], this.container.classList.add("hasBorder");
    }
    const h = s - e, f = a - n, {
      svgFactory: g
    } = this, y = g.createElement("svg");
    y.classList.add("quadrilateralsContainer"), y.setAttribute("width", 0), y.setAttribute("height", 0), y.role = "none";
    const v = g.createElement("defs");
    y.append(v);
    const S = g.createElement("clipPath"), x = `clippath_${this.data.id}`;
    S.setAttribute("id", x), S.setAttribute("clipPathUnits", "objectBoundingBox"), v.append(S);
    for (let _ = 2, k = t.length; _ < k; _ += 8) {
      const T = t[_], R = t[_ + 1], P = t[_ + 2], L = t[_ + 3], D = g.createElement("rect"), N = (P - e) / h, B = (a - R) / f, $ = (T - P) / h, V = (R - L) / f;
      D.setAttribute("x", N), D.setAttribute("y", B), D.setAttribute("width", $), D.setAttribute("height", V), S.append(D), u == null || u.push(`<rect vector-effect="non-scaling-stroke" x="${N}" y="${B}" width="${$}" height="${V}"/>`);
    }
    o(this, Ju) && (u.push("</g></svg>')"), l.backgroundImage = u.join("")), this.container.append(y), this.container.style.clipPath = `url(#${x})`;
  }
  _createPopup(t = null) {
    const {
      data: e
    } = this;
    let n, s;
    t ? (n = {
      str: t.text
    }, s = t.date) : (n = e.contentsObj, s = e.modificationDate);
    const a = w(this, Qi, new _w({
      data: {
        color: e.color,
        titleObj: e.titleObj,
        modificationDate: s,
        contentsObj: n,
        richText: e.richText,
        parentRect: e.rect,
        borderStyle: 0,
        id: `popup_${e.id}`,
        rotation: e.rotation,
        noRotate: !0
      },
      linkService: this.linkService,
      parent: this.parent,
      elements: [this]
    }));
    this.parent._commentManager || this.parent.div.append(a.render());
  }
  get hasPopupElement() {
    return !!(o(this, Qi) || this.popup || this.data.popupRef);
  }
  get extraPopupElement() {
    return o(this, Qi);
  }
  render() {
    ce("Abstract method `AnnotationElement.render` called");
  }
  _getElementsByName(t, e = null) {
    const n = [];
    if (this._fieldObjects) {
      const s = this._fieldObjects[t];
      if (s)
        for (const {
          page: a,
          id: l,
          exportValues: u
        } of s) {
          if (a === -1 || l === e)
            continue;
          const h = typeof u == "string" ? u : null, f = document.querySelector(`[data-element-id="${l}"]`);
          if (f && !hc.has(f)) {
            Rt(`_getElementsByName - element not allowed: ${l}`);
            continue;
          }
          n.push({
            id: l,
            exportValue: h,
            domElement: f
          });
        }
      return n;
    }
    for (const s of document.getElementsByName(t)) {
      const {
        exportValue: a
      } = s, l = s.getAttribute("data-element-id");
      l !== e && hc.has(s) && n.push({
        id: l,
        exportValue: a,
        domElement: s
      });
    }
    return n;
  }
  show() {
    var t;
    this.container && (this.container.hidden = !1), (t = this.popup) == null || t.maybeShow();
  }
  hide() {
    var t;
    this.container && (this.container.hidden = !0), (t = this.popup) == null || t.forceHide();
  }
  getElementsToTriggerPopup() {
    return this.container;
  }
  addHighlightArea() {
    const t = this.getElementsToTriggerPopup();
    if (Array.isArray(t))
      for (const e of t)
        e.classList.add("highlightArea");
    else
      t.classList.add("highlightArea");
  }
  _editOnDoubleClick() {
    if (!this._isEditable)
      return;
    const {
      annotationEditorType: t,
      data: {
        id: e
      }
    } = this;
    this.container.addEventListener("dblclick", () => {
      var n;
      (n = this.linkService.eventBus) == null || n.dispatch("switchannotationeditormode", {
        source: this,
        mode: t,
        editId: e,
        mustEnterInEditMode: !0
      });
    });
  }
  get width() {
    return this.data.rect[2] - this.data.rect[0];
  }
  get height() {
    return this.data.rect[3] - this.data.rect[1];
  }
};
zl = new WeakMap(), Ju = new WeakMap(), Qi = new WeakMap(), sp = new WeakSet(), Ew = function(t) {
  const {
    container: {
      style: e
    },
    data: {
      rect: n,
      rotation: s
    },
    parent: {
      viewport: {
        rawDims: {
          pageWidth: a,
          pageHeight: l,
          pageX: u,
          pageY: h
        }
      }
    }
  } = this;
  n == null || n.splice(0, 4, ...t), e.left = `${100 * (t[0] - u) / a}%`, e.top = `${100 * (l - t[3] + h) / l}%`, s === 0 ? (e.width = `${100 * (t[2] - t[0]) / a}%`, e.height = `${100 * (t[3] - t[1]) / l}%`) : this.setRotation(s);
};
let xe = kA;
class TP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0
    }), this.editor = t.editor;
  }
  render() {
    return this.container.className = "editorAnnotation", this.container;
  }
  createOrUpdatePopup() {
    const {
      editor: t
    } = this;
    t.hasComment && (this._createPopup(t.comment), this.extraPopupElement.popup.renderCommentButton());
  }
  get hasCommentButton() {
    return this.enableComment && this.editor.hasComment;
  }
  get commentButtonPosition() {
    return this.editor.commentButtonPositionInPage;
  }
  get commentText() {
    return this.editor.comment.text;
  }
  set commentText(t) {
    this.editor.comment = t, t || this.removePopup();
  }
  get commentData() {
    return this.editor.getData();
  }
  remove() {
    this.container.remove(), this.container = null, this.removePopup();
  }
}
var Ri, qa, D_, I_;
class wA extends xe {
  constructor(e, n = null) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !!(n != null && n.ignoreBorder),
      createQuadrilaterals: !0
    });
    b(this, Ri);
    this.isTooltipOnly = e.data.isTooltipOnly;
  }
  render() {
    const {
      data: e,
      linkService: n
    } = this, s = document.createElement("a");
    s.setAttribute("data-element-id", e.id);
    let a = !1;
    return e.url ? (n.addLinkAttributes(s, e.url, e.newWindow), a = !0) : e.action ? (this._bindNamedAction(s, e.action, e.overlaidText), a = !0) : e.attachment ? (E(this, Ri, D_).call(this, s, e.attachment, e.overlaidText, e.attachmentDest), a = !0) : e.setOCGState ? (E(this, Ri, I_).call(this, s, e.setOCGState, e.overlaidText), a = !0) : e.dest ? (this._bindLink(s, e.dest, e.overlaidText), a = !0) : (e.actions && (e.actions.Action || e.actions["Mouse Up"] || e.actions["Mouse Down"]) && this.enableScripting && this.hasJSActions && (this._bindJSAction(s, e), a = !0), e.resetForm ? (this._bindResetFormAction(s, e.resetForm), a = !0) : this.isTooltipOnly && !a && (this._bindLink(s, ""), a = !0)), this.container.classList.add("linkAnnotation"), a && this.container.append(s), this.container;
  }
  _bindLink(e, n, s = "") {
    e.href = this.linkService.getDestinationHash(n), e.onclick = () => (n && this.linkService.goToDestination(n), !1), (n || n === "") && E(this, Ri, qa).call(this), s && (e.title = s);
  }
  _bindNamedAction(e, n, s = "") {
    e.href = this.linkService.getAnchorUrl(""), e.onclick = () => (this.linkService.executeNamedAction(n), !1), s && (e.title = s), E(this, Ri, qa).call(this);
  }
  _bindJSAction(e, n) {
    e.href = this.linkService.getAnchorUrl("");
    const s = /* @__PURE__ */ new Map([["Action", "onclick"], ["Mouse Up", "onmouseup"], ["Mouse Down", "onmousedown"]]);
    for (const a of Object.keys(n.actions)) {
      const l = s.get(a);
      l && (e[l] = () => {
        var u;
        return (u = this.linkService.eventBus) == null || u.dispatch("dispatcheventinsandbox", {
          source: this,
          detail: {
            id: n.id,
            name: a
          }
        }), !1;
      });
    }
    n.overlaidText && (e.title = n.overlaidText), e.onclick || (e.onclick = () => !1), E(this, Ri, qa).call(this);
  }
  _bindResetFormAction(e, n) {
    const s = e.onclick;
    if (s || (e.href = this.linkService.getAnchorUrl("")), E(this, Ri, qa).call(this), !this._fieldObjects) {
      Rt('_bindResetFormAction - "resetForm" action not supported, ensure that the `fieldObjects` parameter is provided.'), s || (e.onclick = () => !1);
      return;
    }
    e.onclick = () => {
      var y;
      s == null || s();
      const {
        fields: a,
        refs: l,
        include: u
      } = n, h = [];
      if (a.length !== 0 || l.length !== 0) {
        const v = new Set(l);
        for (const S of a) {
          const x = this._fieldObjects[S] || [];
          for (const {
            id: _
          } of x)
            v.add(_);
        }
        for (const S of Object.values(this._fieldObjects))
          for (const x of S)
            v.has(x.id) === u && h.push(x);
      } else
        for (const v of Object.values(this._fieldObjects))
          h.push(...v);
      const f = this.annotationStorage, g = [];
      for (const v of h) {
        const {
          id: S
        } = v;
        switch (g.push(S), v.type) {
          case "text": {
            const _ = v.defaultValue || "";
            f.setValue(S, {
              value: _
            });
            break;
          }
          case "checkbox":
          case "radiobutton": {
            const _ = v.defaultValue === v.exportValues;
            f.setValue(S, {
              value: _
            });
            break;
          }
          case "combobox":
          case "listbox": {
            const _ = v.defaultValue || "";
            f.setValue(S, {
              value: _
            });
            break;
          }
          default:
            continue;
        }
        const x = document.querySelector(`[data-element-id="${S}"]`);
        if (x) {
          if (!hc.has(x)) {
            Rt(`_bindResetFormAction - element not allowed: ${S}`);
            continue;
          }
        } else continue;
        x.dispatchEvent(new Event("resetform"));
      }
      return this.enableScripting && ((y = this.linkService.eventBus) == null || y.dispatch("dispatcheventinsandbox", {
        source: this,
        detail: {
          id: "app",
          ids: g,
          name: "ResetForm"
        }
      })), !1;
    };
  }
}
Ri = new WeakSet(), qa = function() {
  this.container.setAttribute("data-internal-link", "");
}, D_ = function(e, n, s = "", a = null) {
  e.href = this.linkService.getAnchorUrl(""), n.description ? e.title = n.description : s && (e.title = s), e.onclick = () => {
    var l;
    return (l = this.downloadManager) == null || l.openOrDownloadData(n.content, n.filename, a), !1;
  }, E(this, Ri, qa).call(this);
}, I_ = function(e, n, s = "") {
  e.href = this.linkService.getAnchorUrl(""), e.onclick = () => (this.linkService.executeSetOCGState(n), !1), s && (e.title = s), E(this, Ri, qa).call(this);
};
class kP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0
    });
  }
  render() {
    this.container.classList.add("textAnnotation");
    const t = document.createElement("img");
    return t.src = this.imageResourcesPath + "annotation-" + this.data.name.toLowerCase() + ".svg", t.setAttribute("data-l10n-id", "pdfjs-text-annotation-type"), t.setAttribute("data-l10n-args", JSON.stringify({
      type: this.data.name
    })), !this.data.popupRef && this.hasPopupData && this._createPopup(), this.container.append(t), this.container;
  }
}
class pc extends xe {
  render() {
    return this.container;
  }
  showElementAndHideCanvas(t) {
    var e;
    this.data.hasOwnCanvas && (((e = t.previousSibling) == null ? void 0 : e.nodeName) === "CANVAS" && (t.previousSibling.hidden = !0), t.hidden = !1);
  }
  _getKeyModifier(t) {
    return cn.platform.isMac ? t.metaKey : t.ctrlKey;
  }
  _setEventListener(t, e, n, s, a) {
    n.includes("mouse") ? t.addEventListener(n, (l) => {
      var u;
      (u = this.linkService.eventBus) == null || u.dispatch("dispatcheventinsandbox", {
        source: this,
        detail: {
          id: this.data.id,
          name: s,
          value: a(l),
          shift: l.shiftKey,
          modifier: this._getKeyModifier(l)
        }
      });
    }) : t.addEventListener(n, (l) => {
      var u;
      if (n === "blur") {
        if (!e.focused || !l.relatedTarget)
          return;
        e.focused = !1;
      } else if (n === "focus") {
        if (e.focused)
          return;
        e.focused = !0;
      }
      a && ((u = this.linkService.eventBus) == null || u.dispatch("dispatcheventinsandbox", {
        source: this,
        detail: {
          id: this.data.id,
          name: s,
          value: a(l)
        }
      }));
    });
  }
  _setEventListeners(t, e, n, s) {
    var a, l, u;
    for (const [h, f] of n)
      (f === "Action" || (a = this.data.actions) != null && a[f]) && ((f === "Focus" || f === "Blur") && (e || (e = {
        focused: !1
      })), this._setEventListener(t, e, h, f, s), f === "Focus" && !((l = this.data.actions) != null && l.Blur) ? this._setEventListener(t, e, "blur", "Blur", null) : f === "Blur" && !((u = this.data.actions) != null && u.Focus) && this._setEventListener(t, e, "focus", "Focus", null));
  }
  _setBackgroundColor(t) {
    const e = this.data.backgroundColor || null;
    t.style.backgroundColor = e === null ? "transparent" : ft.makeHexColor(e[0], e[1], e[2]);
  }
  _setTextStyle(t) {
    const e = ["left", "center", "right"], {
      fontColor: n
    } = this.data.defaultAppearanceData, s = this.data.defaultAppearanceData.fontSize || CP, a = t.style;
    let l;
    const u = 2, h = (f) => Math.round(10 * f) / 10;
    if (this.data.multiLine) {
      const f = Math.abs(this.data.rect[3] - this.data.rect[1] - u), g = Math.round(f / (_0 * s)) || 1, y = f / g;
      l = Math.min(s, h(y / _0));
    } else {
      const f = Math.abs(this.data.rect[3] - this.data.rect[1] - u);
      l = Math.min(s, h(f / _0));
    }
    a.fontSize = `calc(${l}px * var(--total-scale-factor))`, a.color = ft.makeHexColor(n[0], n[1], n[2]), this.data.textAlignment !== null && (a.textAlign = e[this.data.textAlignment]);
  }
  _setRequired(t, e) {
    e ? t.setAttribute("required", !0) : t.removeAttribute("required"), t.setAttribute("aria-required", e);
  }
}
class PP extends pc {
  constructor(t) {
    const e = t.renderForms || t.data.hasOwnCanvas || !t.data.hasAppearance && !!t.data.fieldValue;
    super(t, {
      isRenderable: e
    });
  }
  setPropertyOnSiblings(t, e, n, s) {
    const a = this.annotationStorage;
    for (const l of this._getElementsByName(t.name, t.id))
      l.domElement && (l.domElement[e] = n), a.setValue(l.id, {
        [s]: n
      });
  }
  render() {
    var s, a;
    const t = this.annotationStorage, e = this.data.id;
    this.container.classList.add("textWidgetAnnotation");
    let n = null;
    if (this.renderForms) {
      const l = t.getValue(e, {
        value: this.data.fieldValue
      });
      let u = l.value || "";
      const h = t.getValue(e, {
        charLimit: this.data.maxLen
      }).charLimit;
      h && u.length > h && (u = u.slice(0, h));
      let f = l.formattedValue || ((s = this.data.textContent) == null ? void 0 : s.join(`
`)) || null;
      f && this.data.comb && (f = f.replaceAll(/\s+/g, ""));
      const g = {
        userValue: u,
        formattedValue: f,
        lastCommittedValue: null,
        commitKey: 1,
        focused: !1
      };
      this.data.multiLine ? (n = document.createElement("textarea"), n.textContent = f ?? u, this.data.doNotScroll && (n.style.overflowY = "hidden")) : (n = document.createElement("input"), n.type = this.data.password ? "password" : "text", n.setAttribute("value", f ?? u), this.data.doNotScroll && (n.style.overflowX = "hidden")), this.data.hasOwnCanvas && (n.hidden = !0), hc.add(n), n.setAttribute("data-element-id", e), n.disabled = this.data.readOnly, n.name = this.data.fieldName, n.tabIndex = 0;
      const {
        datetimeFormat: y,
        datetimeType: v,
        timeStep: S
      } = this.data, x = !!v && this.enableScripting;
      y && (n.title = y), this._setRequired(n, this.data.required), h && (n.maxLength = h), n.addEventListener("input", (k) => {
        t.setValue(e, {
          value: k.target.value
        }), this.setPropertyOnSiblings(n, "value", k.target.value, "value"), g.formattedValue = null;
      }), n.addEventListener("resetform", (k) => {
        const T = this.data.defaultFieldValue ?? "";
        n.value = g.userValue = T, g.formattedValue = null;
      });
      let _ = (k) => {
        const {
          formattedValue: T
        } = g;
        T != null && (k.target.value = T), k.target.scrollLeft = 0;
      };
      if (this.enableScripting && this.hasJSActions) {
        n.addEventListener("focus", (T) => {
          var P;
          if (g.focused)
            return;
          const {
            target: R
          } = T;
          if (x && (R.type = v, S && (R.step = S)), g.userValue) {
            const L = g.userValue;
            if (x)
              if (v === "time") {
                const D = new Date(L), N = [D.getHours(), D.getMinutes(), D.getSeconds()];
                R.value = N.map((B) => B.toString().padStart(2, "0")).join(":");
              } else
                R.value = new Date(L - xP).toISOString().split(v === "date" ? "T" : ".", 1)[0];
            else
              R.value = L;
          }
          g.lastCommittedValue = R.value, g.commitKey = 1, (P = this.data.actions) != null && P.Focus || (g.focused = !0);
        }), n.addEventListener("updatefromsandbox", (T) => {
          this.showElementAndHideCanvas(T.target);
          const R = {
            value(P) {
              g.userValue = P.detail.value ?? "", x || t.setValue(e, {
                value: g.userValue.toString()
              }), P.target.value = g.userValue;
            },
            formattedValue(P) {
              const {
                formattedValue: L
              } = P.detail;
              g.formattedValue = L, L != null && P.target !== document.activeElement && (P.target.value = L);
              const D = {
                formattedValue: L
              };
              x && (D.value = L), t.setValue(e, D);
            },
            selRange(P) {
              P.target.setSelectionRange(...P.detail.selRange);
            },
            charLimit: (P) => {
              var B;
              const {
                charLimit: L
              } = P.detail, {
                target: D
              } = P;
              if (L === 0) {
                D.removeAttribute("maxLength");
                return;
              }
              D.setAttribute("maxLength", L);
              let N = g.userValue;
              !N || N.length <= L || (N = N.slice(0, L), D.value = g.userValue = N, t.setValue(e, {
                value: N
              }), (B = this.linkService.eventBus) == null || B.dispatch("dispatcheventinsandbox", {
                source: this,
                detail: {
                  id: e,
                  name: "Keystroke",
                  value: N,
                  willCommit: !0,
                  commitKey: 1,
                  selStart: D.selectionStart,
                  selEnd: D.selectionEnd
                }
              }));
            }
          };
          this._dispatchEventFromSandbox(R, T);
        }), n.addEventListener("keydown", (T) => {
          var L;
          g.commitKey = 1;
          let R = -1;
          if (T.key === "Escape" ? R = 0 : T.key === "Enter" && !this.data.multiLine ? R = 2 : T.key === "Tab" && (g.commitKey = 3), R === -1)
            return;
          const {
            value: P
          } = T.target;
          g.lastCommittedValue !== P && (g.lastCommittedValue = P, g.userValue = P, (L = this.linkService.eventBus) == null || L.dispatch("dispatcheventinsandbox", {
            source: this,
            detail: {
              id: e,
              name: "Keystroke",
              value: P,
              willCommit: !0,
              commitKey: R,
              selStart: T.target.selectionStart,
              selEnd: T.target.selectionEnd
            }
          }));
        });
        const k = _;
        _ = null, n.addEventListener("blur", (T) => {
          var L, D;
          if (!g.focused || !T.relatedTarget)
            return;
          (L = this.data.actions) != null && L.Blur || (g.focused = !1);
          const {
            target: R
          } = T;
          let {
            value: P
          } = R;
          if (x) {
            if (P && v === "time") {
              const N = P.split(":").map((B) => parseInt(B, 10));
              P = new Date(2e3, 0, 1, N[0], N[1], N[2] || 0).valueOf(), R.step = "";
            } else
              P.includes("T") || (P = `${P}T00:00`), P = new Date(P).valueOf();
            R.type = "text";
          }
          g.userValue = P, g.lastCommittedValue !== P && ((D = this.linkService.eventBus) == null || D.dispatch("dispatcheventinsandbox", {
            source: this,
            detail: {
              id: e,
              name: "Keystroke",
              value: P,
              willCommit: !0,
              commitKey: g.commitKey,
              selStart: T.target.selectionStart,
              selEnd: T.target.selectionEnd
            }
          })), k(T);
        }), (a = this.data.actions) != null && a.Keystroke && n.addEventListener("beforeinput", (T) => {
          var V;
          g.lastCommittedValue = null;
          const {
            data: R,
            target: P
          } = T, {
            value: L,
            selectionStart: D,
            selectionEnd: N
          } = P;
          let B = D, $ = N;
          switch (T.inputType) {
            case "deleteWordBackward": {
              const X = L.substring(0, D).match(/\w*[^\w]*$/);
              X && (B -= X[0].length);
              break;
            }
            case "deleteWordForward": {
              const X = L.substring(D).match(/^[^\w]*\w*/);
              X && ($ += X[0].length);
              break;
            }
            case "deleteContentBackward":
              D === N && (B -= 1);
              break;
            case "deleteContentForward":
              D === N && ($ += 1);
              break;
          }
          T.preventDefault(), (V = this.linkService.eventBus) == null || V.dispatch("dispatcheventinsandbox", {
            source: this,
            detail: {
              id: e,
              name: "Keystroke",
              value: L,
              change: R || "",
              willCommit: !1,
              selStart: B,
              selEnd: $
            }
          });
        }), this._setEventListeners(n, g, [["focus", "Focus"], ["blur", "Blur"], ["mousedown", "Mouse Down"], ["mouseenter", "Mouse Enter"], ["mouseleave", "Mouse Exit"], ["mouseup", "Mouse Up"]], (T) => T.target.value);
      }
      if (_ && n.addEventListener("blur", _), this.data.comb) {
        const T = (this.data.rect[2] - this.data.rect[0]) / h;
        n.classList.add("comb"), n.style.letterSpacing = `calc(${T}px * var(--total-scale-factor) - 1ch)`;
      }
    } else
      n = document.createElement("div"), n.textContent = this.data.fieldValue, n.style.verticalAlign = "middle", n.style.display = "table-cell", this.data.hasOwnCanvas && (n.hidden = !0);
    return this._setTextStyle(n), this._setBackgroundColor(n), this._setDefaultPropertiesFromJS(n), this.container.append(n), this.container;
  }
}
class RP extends pc {
  constructor(t) {
    super(t, {
      isRenderable: !!t.data.hasOwnCanvas
    });
  }
}
class MP extends pc {
  constructor(t) {
    super(t, {
      isRenderable: t.renderForms
    });
  }
  render() {
    const t = this.annotationStorage, e = this.data, n = e.id;
    let s = t.getValue(n, {
      value: e.exportValue === e.fieldValue
    }).value;
    typeof s == "string" && (s = s !== "Off", t.setValue(n, {
      value: s
    })), this.container.classList.add("buttonWidgetAnnotation", "checkBox");
    const a = document.createElement("input");
    return hc.add(a), a.setAttribute("data-element-id", n), a.disabled = e.readOnly, this._setRequired(a, this.data.required), a.type = "checkbox", a.name = e.fieldName, s && a.setAttribute("checked", !0), a.setAttribute("exportValue", e.exportValue), a.tabIndex = 0, a.addEventListener("change", (l) => {
      const {
        name: u,
        checked: h
      } = l.target;
      for (const f of this._getElementsByName(u, n)) {
        const g = h && f.exportValue === e.exportValue;
        f.domElement && (f.domElement.checked = g), t.setValue(f.id, {
          value: g
        });
      }
      t.setValue(n, {
        value: h
      });
    }), a.addEventListener("resetform", (l) => {
      const u = e.defaultFieldValue || "Off";
      l.target.checked = u === e.exportValue;
    }), this.enableScripting && this.hasJSActions && (a.addEventListener("updatefromsandbox", (l) => {
      const u = {
        value(h) {
          h.target.checked = h.detail.value !== "Off", t.setValue(n, {
            value: h.target.checked
          });
        }
      };
      this._dispatchEventFromSandbox(u, l);
    }), this._setEventListeners(a, null, [["change", "Validate"], ["change", "Action"], ["focus", "Focus"], ["blur", "Blur"], ["mousedown", "Mouse Down"], ["mouseenter", "Mouse Enter"], ["mouseleave", "Mouse Exit"], ["mouseup", "Mouse Up"]], (l) => l.target.checked)), this._setBackgroundColor(a), this._setDefaultPropertiesFromJS(a), this.container.append(a), this.container;
  }
}
class F_ extends pc {
  constructor(t) {
    super(t, {
      isRenderable: t.renderForms
    });
  }
  render() {
    this.container.classList.add("buttonWidgetAnnotation", "radioButton");
    const t = this.annotationStorage, e = this.data, n = e.id;
    let s = t.getValue(n, {
      value: e.fieldValue === e.buttonValue
    }).value;
    if (typeof s == "string" && (s = s !== e.buttonValue, t.setValue(n, {
      value: s
    })), s)
      for (const l of this._getElementsByName(e.fieldName, n))
        t.setValue(l.id, {
          value: !1
        });
    const a = document.createElement("input");
    if (hc.add(a), a.setAttribute("data-element-id", n), a.disabled = e.readOnly, this._setRequired(a, this.data.required), a.type = "radio", a.name = e.fieldName, s && a.setAttribute("checked", !0), a.tabIndex = 0, a.addEventListener("change", (l) => {
      const {
        name: u,
        checked: h
      } = l.target;
      for (const f of this._getElementsByName(u, n))
        t.setValue(f.id, {
          value: !1
        });
      t.setValue(n, {
        value: h
      });
    }), a.addEventListener("resetform", (l) => {
      const u = e.defaultFieldValue;
      l.target.checked = u != null && u === e.buttonValue;
    }), this.enableScripting && this.hasJSActions) {
      const l = e.buttonValue;
      a.addEventListener("updatefromsandbox", (u) => {
        const h = {
          value: (f) => {
            const g = l === f.detail.value;
            for (const y of this._getElementsByName(f.target.name)) {
              const v = g && y.id === n;
              y.domElement && (y.domElement.checked = v), t.setValue(y.id, {
                value: v
              });
            }
          }
        };
        this._dispatchEventFromSandbox(h, u);
      }), this._setEventListeners(a, null, [["change", "Validate"], ["change", "Action"], ["focus", "Focus"], ["blur", "Blur"], ["mousedown", "Mouse Down"], ["mouseenter", "Mouse Enter"], ["mouseleave", "Mouse Exit"], ["mouseup", "Mouse Up"]], (u) => u.target.checked);
    }
    return this._setBackgroundColor(a), this._setDefaultPropertiesFromJS(a), this.container.append(a), this.container;
  }
}
class LP extends wA {
  constructor(t) {
    super(t, {
      ignoreBorder: t.data.hasAppearance
    });
  }
  render() {
    const t = super.render();
    t.classList.add("buttonWidgetAnnotation", "pushButton");
    const e = t.lastChild;
    return this.enableScripting && this.hasJSActions && e && (this._setDefaultPropertiesFromJS(e), e.addEventListener("updatefromsandbox", (n) => {
      this._dispatchEventFromSandbox({}, n);
    })), t;
  }
}
class DP extends pc {
  constructor(t) {
    super(t, {
      isRenderable: t.renderForms
    });
  }
  render() {
    this.container.classList.add("choiceWidgetAnnotation");
    const t = this.annotationStorage, e = this.data.id, n = t.getValue(e, {
      value: this.data.fieldValue
    }), s = document.createElement("select");
    hc.add(s), s.setAttribute("data-element-id", e), s.disabled = this.data.readOnly, this._setRequired(s, this.data.required), s.name = this.data.fieldName, s.tabIndex = 0;
    let a = this.data.combo && this.data.options.length > 0;
    this.data.combo || (s.size = this.data.options.length, this.data.multiSelect && (s.multiple = !0)), s.addEventListener("resetform", (g) => {
      const y = this.data.defaultFieldValue;
      for (const v of s.options)
        v.selected = v.value === y;
    });
    for (const g of this.data.options) {
      const y = document.createElement("option");
      y.textContent = g.displayValue, y.value = g.exportValue, n.value.includes(g.exportValue) && (y.setAttribute("selected", !0), a = !1), s.append(y);
    }
    let l = null;
    if (a) {
      const g = document.createElement("option");
      g.value = " ", g.setAttribute("hidden", !0), g.setAttribute("selected", !0), s.prepend(g), l = () => {
        g.remove(), s.removeEventListener("input", l), l = null;
      }, s.addEventListener("input", l);
    }
    const u = (g) => {
      const y = g ? "value" : "textContent", {
        options: v,
        multiple: S
      } = s;
      return S ? Array.prototype.filter.call(v, (x) => x.selected).map((x) => x[y]) : v.selectedIndex === -1 ? null : v[v.selectedIndex][y];
    };
    let h = u(!1);
    const f = (g) => {
      const y = g.target.options;
      return Array.prototype.map.call(y, (v) => ({
        displayValue: v.textContent,
        exportValue: v.value
      }));
    };
    return this.enableScripting && this.hasJSActions ? (s.addEventListener("updatefromsandbox", (g) => {
      const y = {
        value(v) {
          l == null || l();
          const S = v.detail.value, x = new Set(Array.isArray(S) ? S : [S]);
          for (const _ of s.options)
            _.selected = x.has(_.value);
          t.setValue(e, {
            value: u(!0)
          }), h = u(!1);
        },
        multipleSelection(v) {
          s.multiple = !0;
        },
        remove(v) {
          const S = s.options, x = v.detail.remove;
          S[x].selected = !1, s.remove(x), S.length > 0 && Array.prototype.findIndex.call(S, (k) => k.selected) === -1 && (S[0].selected = !0), t.setValue(e, {
            value: u(!0),
            items: f(v)
          }), h = u(!1);
        },
        clear(v) {
          for (; s.length !== 0; )
            s.remove(0);
          t.setValue(e, {
            value: null,
            items: []
          }), h = u(!1);
        },
        insert(v) {
          const {
            index: S,
            displayValue: x,
            exportValue: _
          } = v.detail.insert, k = s.children[S], T = document.createElement("option");
          T.textContent = x, T.value = _, k ? k.before(T) : s.append(T), t.setValue(e, {
            value: u(!0),
            items: f(v)
          }), h = u(!1);
        },
        items(v) {
          const {
            items: S
          } = v.detail;
          for (; s.length !== 0; )
            s.remove(0);
          for (const x of S) {
            const {
              displayValue: _,
              exportValue: k
            } = x, T = document.createElement("option");
            T.textContent = _, T.value = k, s.append(T);
          }
          s.options.length > 0 && (s.options[0].selected = !0), t.setValue(e, {
            value: u(!0),
            items: f(v)
          }), h = u(!1);
        },
        indices(v) {
          const S = new Set(v.detail.indices);
          for (const x of v.target.options)
            x.selected = S.has(x.index);
          t.setValue(e, {
            value: u(!0)
          }), h = u(!1);
        },
        editable(v) {
          v.target.disabled = !v.detail.editable;
        }
      };
      this._dispatchEventFromSandbox(y, g);
    }), s.addEventListener("input", (g) => {
      var S;
      const y = u(!0), v = u(!1);
      t.setValue(e, {
        value: y
      }), g.preventDefault(), (S = this.linkService.eventBus) == null || S.dispatch("dispatcheventinsandbox", {
        source: this,
        detail: {
          id: e,
          name: "Keystroke",
          value: h,
          change: v,
          changeEx: y,
          willCommit: !1,
          commitKey: 1,
          keyDown: !1
        }
      });
    }), this._setEventListeners(s, null, [["focus", "Focus"], ["blur", "Blur"], ["mousedown", "Mouse Down"], ["mouseenter", "Mouse Enter"], ["mouseleave", "Mouse Exit"], ["mouseup", "Mouse Up"], ["input", "Action"], ["input", "Validate"]], (g) => g.target.value)) : s.addEventListener("input", function(g) {
      t.setValue(e, {
        value: u(!0)
      });
    }), this.data.combo && this._setTextStyle(s), this._setBackgroundColor(s), this._setDefaultPropertiesFromJS(s), this.container.append(s), this.container;
  }
}
var rp, Cw;
class _w extends xe {
  constructor(e) {
    const {
      data: n,
      elements: s,
      parent: a
    } = e, l = !!a._commentManager;
    super(e, {
      isRenderable: !l && xe._hasPopupData(n)
    });
    b(this, rp);
    if (this.elements = s, l && xe._hasPopupData(n)) {
      const u = this.popup = E(this, rp, Cw).call(this);
      for (const h of s)
        h.popup = u;
    } else
      this.popup = null;
  }
  render() {
    const {
      container: e
    } = this;
    e.classList.add("popupAnnotation"), e.role = "comment";
    const n = this.popup = E(this, rp, Cw).call(this), s = [];
    for (const a of this.elements)
      a.popup = n, a.container.ariaHasPopup = "dialog", s.push(a.data.id), a.addHighlightArea();
    return this.container.setAttribute("aria-controls", s.map((a) => `${hA}${a}`).join(",")), this.container;
  }
}
rp = new WeakSet(), Cw = function() {
  return new IP({
    container: this.container,
    color: this.data.color,
    titleObj: this.data.titleObj,
    modificationDate: this.data.modificationDate || this.data.creationDate,
    contentsObj: this.data.contentsObj,
    richText: this.data.richText,
    rect: this.data.rect,
    parentRect: this.data.parentRect || null,
    parent: this.parent,
    elements: this.elements,
    open: this.data.open,
    commentManager: this.parent._commentManager
  });
};
var yi, aa, fy, py, th, eh, Me, tr, la, Ul, nh, ih, er, vi, qr, Yr, nn, Kr, ca, op, Qr, sh, Hl, ua, Rn, ha, Nt, _m, xw, Tw, kw, Cm, Pw, N_, O_, B_, z_, xm, Tm, Rw;
class IP {
  constructor({
    container: t,
    color: e,
    elements: n,
    titleObj: s,
    modificationDate: a,
    contentsObj: l,
    richText: u,
    parent: h,
    rect: f,
    parentRect: g,
    open: y,
    commentManager: v = null
  }) {
    b(this, Nt);
    b(this, yi, null);
    b(this, aa, E(this, Nt, B_).bind(this));
    b(this, fy, E(this, Nt, Rw).bind(this));
    b(this, py, E(this, Nt, Tm).bind(this));
    b(this, th, E(this, Nt, xm).bind(this));
    b(this, eh, null);
    b(this, Me, null);
    b(this, tr, null);
    b(this, la, null);
    b(this, Ul, null);
    b(this, nh, null);
    b(this, ih, null);
    b(this, er, !1);
    b(this, vi, null);
    b(this, qr, null);
    b(this, Yr, null);
    b(this, nn, null);
    b(this, Kr, null);
    b(this, ca, null);
    b(this, op, null);
    b(this, Qr, null);
    b(this, sh, null);
    b(this, Hl, null);
    b(this, ua, !1);
    b(this, Rn, null);
    b(this, ha, null);
    w(this, Me, t), w(this, sh, s), w(this, tr, l), w(this, Qr, u), w(this, nh, h), w(this, eh, e), w(this, op, f), w(this, ih, g), w(this, Ul, n), w(this, yi, v), w(this, Rn, n[0]), w(this, la, qd.toDateObject(a)), this.trigger = n.flatMap((S) => S.getElementsToTriggerPopup()), v ? this.renderCommentButton() : (E(this, Nt, _m).call(this), o(this, Me).hidden = !0, y && E(this, Nt, xm).call(this));
  }
  renderCommentButton() {
    if (o(this, nn) || (o(this, Kr) || E(this, Nt, xw).call(this), !o(this, Kr)))
      return;
    const {
      signal: t
    } = w(this, qr, new AbortController()), e = !!o(this, Rn).extraPopupElement, n = () => {
      o(this, yi).toggleCommentPopup(this, !0, void 0, !e);
    }, s = () => {
      o(this, yi).toggleCommentPopup(this, !1, !0, !e);
    }, a = () => {
      o(this, yi).toggleCommentPopup(this, !1, !1);
    };
    if (e) {
      w(this, nn, o(this, Rn).container);
      for (const l of this.trigger)
        l.ariaHasPopup = "dialog", l.ariaControls = "commentPopup", l.addEventListener("keydown", o(this, aa), {
          signal: t
        }), l.addEventListener("click", n, {
          signal: t
        }), l.addEventListener("pointerenter", s, {
          signal: t
        }), l.addEventListener("pointerleave", a, {
          signal: t
        }), l.classList.add("popupTriggerArea");
    } else {
      const l = w(this, nn, document.createElement("button"));
      l.className = "annotationCommentButton";
      const u = o(this, Rn).container;
      l.style.zIndex = u.style.zIndex + 1, l.tabIndex = 0, l.ariaHasPopup = "dialog", l.ariaControls = "commentPopup", l.setAttribute("data-l10n-id", "pdfjs-show-comment-button"), E(this, Nt, kw).call(this), E(this, Nt, Tw).call(this), l.addEventListener("keydown", o(this, aa), {
        signal: t
      }), l.addEventListener("click", n, {
        signal: t
      }), l.addEventListener("pointerenter", s, {
        signal: t
      }), l.addEventListener("pointerleave", a, {
        signal: t
      }), u.after(l);
    }
  }
  get commentButtonColor() {
    const {
      color: t,
      opacity: e
    } = o(this, Rn).commentData;
    return t ? o(this, nh)._commentManager.makeCommentColor(t, e) : null;
  }
  focusCommentButton() {
    setTimeout(() => {
      var t;
      (t = o(this, nn)) == null || t.focus();
    }, 0);
  }
  getData() {
    const {
      richText: t,
      color: e,
      opacity: n,
      creationDate: s,
      modificationDate: a
    } = o(this, Rn).commentData;
    return {
      contentsObj: {
        str: this.comment
      },
      richText: t,
      color: e,
      opacity: n,
      creationDate: s,
      modificationDate: a
    };
  }
  get elementBeforePopup() {
    return o(this, nn);
  }
  get comment() {
    return o(this, ha) || w(this, ha, o(this, Rn).commentText), o(this, ha);
  }
  set comment(t) {
    t !== this.comment && (o(this, Rn).commentText = w(this, ha, t));
  }
  get parentBoundingClientRect() {
    return o(this, Rn).layer.getBoundingClientRect();
  }
  setCommentButtonStates({
    selected: t,
    hasPopup: e
  }) {
    o(this, nn) && (o(this, nn).classList.toggle("selected", t), o(this, nn).ariaExpanded = e);
  }
  setSelectedCommentButton(t) {
    o(this, nn).classList.toggle("selected", t);
  }
  get commentPopupPosition() {
    if (o(this, ca))
      return o(this, ca);
    const {
      x: t,
      y: e,
      height: n
    } = o(this, nn).getBoundingClientRect(), {
      x: s,
      y: a,
      width: l,
      height: u
    } = o(this, Rn).layer.getBoundingClientRect();
    return [(t - s) / l, (e + n - a) / u];
  }
  set commentPopupPosition(t) {
    w(this, ca, t);
  }
  hasDefaultPopupPosition() {
    return o(this, ca) === null;
  }
  get commentButtonPosition() {
    return o(this, Kr);
  }
  get commentButtonWidth() {
    return o(this, nn).getBoundingClientRect().width / this.parentBoundingClientRect.width;
  }
  editComment(t) {
    const [e, n] = o(this, ca) || this.commentButtonPosition.map((f) => f / 100), s = this.parentBoundingClientRect, {
      x: a,
      y: l,
      width: u,
      height: h
    } = s;
    o(this, yi).showDialog(null, this, a + e * u, l + n * h, {
      ...t,
      parentDimensions: s
    });
  }
  render() {
    var n, s;
    if (o(this, vi))
      return;
    const t = w(this, vi, document.createElement("div"));
    if (t.className = "popup", o(this, eh)) {
      const a = t.style.outlineColor = ft.makeHexColor(...o(this, eh));
      t.style.backgroundColor = `color-mix(in srgb, ${a} 30%, white)`;
    }
    const e = document.createElement("span");
    if (e.className = "header", (n = o(this, sh)) != null && n.str) {
      const a = document.createElement("span");
      a.className = "title", e.append(a), {
        dir: a.dir,
        str: a.textContent
      } = o(this, sh);
    }
    if (t.append(e), o(this, la)) {
      const a = document.createElement("time");
      a.className = "popupDate", a.setAttribute("data-l10n-id", "pdfjs-annotation-date-time-string"), a.setAttribute("data-l10n-args", JSON.stringify({
        dateObj: o(this, la).valueOf()
      })), a.dateTime = o(this, la).toISOString(), e.append(a);
    }
    fA({
      html: o(this, Nt, Cm) || o(this, tr).str,
      dir: (s = o(this, tr)) == null ? void 0 : s.dir,
      className: "popupContent"
    }, t), o(this, Me).append(t);
  }
  updateEdited({
    rect: t,
    popup: e,
    deleted: n
  }) {
    var s;
    if (o(this, yi)) {
      n ? (this.remove(), w(this, ha, null)) : e && (e.deleted ? this.remove() : (E(this, Nt, kw).call(this), w(this, ha, e.text))), t && (w(this, Kr, null), E(this, Nt, xw).call(this), E(this, Nt, Tw).call(this));
      return;
    }
    if (n || e != null && e.deleted) {
      this.remove();
      return;
    }
    E(this, Nt, _m).call(this), o(this, Hl) || w(this, Hl, {
      contentsObj: o(this, tr),
      richText: o(this, Qr)
    }), t && w(this, Yr, null), e && e.text && (w(this, Qr, E(this, Nt, O_).call(this, e.text)), w(this, la, qd.toDateObject(e.date)), w(this, tr, null)), (s = o(this, vi)) == null || s.remove(), w(this, vi, null);
  }
  resetEdited() {
    var t;
    o(this, Hl) && ({
      contentsObj: Ge(this, tr)._,
      richText: Ge(this, Qr)._
    } = o(this, Hl), w(this, Hl, null), (t = o(this, vi)) == null || t.remove(), w(this, vi, null), w(this, Yr, null));
  }
  remove() {
    var t, e, n;
    if ((t = o(this, qr)) == null || t.abort(), w(this, qr, null), (e = o(this, vi)) == null || e.remove(), w(this, vi, null), w(this, ua, !1), w(this, er, !1), (n = o(this, nn)) == null || n.remove(), w(this, nn, null), this.trigger)
      for (const s of this.trigger)
        s.classList.remove("popupTriggerArea");
  }
  forceHide() {
    w(this, ua, this.isVisible), o(this, ua) && (o(this, Me).hidden = !0);
  }
  maybeShow() {
    o(this, yi) || (E(this, Nt, _m).call(this), o(this, ua) && (o(this, vi) || E(this, Nt, Tm).call(this), w(this, ua, !1), o(this, Me).hidden = !1));
  }
  get isVisible() {
    return o(this, yi) ? !1 : o(this, Me).hidden === !1;
  }
}
yi = new WeakMap(), aa = new WeakMap(), fy = new WeakMap(), py = new WeakMap(), th = new WeakMap(), eh = new WeakMap(), Me = new WeakMap(), tr = new WeakMap(), la = new WeakMap(), Ul = new WeakMap(), nh = new WeakMap(), ih = new WeakMap(), er = new WeakMap(), vi = new WeakMap(), qr = new WeakMap(), Yr = new WeakMap(), nn = new WeakMap(), Kr = new WeakMap(), ca = new WeakMap(), op = new WeakMap(), Qr = new WeakMap(), sh = new WeakMap(), Hl = new WeakMap(), ua = new WeakMap(), Rn = new WeakMap(), ha = new WeakMap(), Nt = new WeakSet(), _m = function() {
  var e;
  if (o(this, qr))
    return;
  w(this, qr, new AbortController());
  const {
    signal: t
  } = o(this, qr);
  for (const n of this.trigger)
    n.addEventListener("click", o(this, th), {
      signal: t
    }), n.addEventListener("pointerenter", o(this, py), {
      signal: t
    }), n.addEventListener("pointerleave", o(this, fy), {
      signal: t
    }), n.classList.add("popupTriggerArea");
  for (const n of o(this, Ul))
    (e = n.container) == null || e.addEventListener("keydown", o(this, aa), {
      signal: t
    });
}, xw = function() {
  const t = o(this, Ul).find((e) => e.hasCommentButton);
  t && w(this, Kr, t._normalizePoint(t.commentButtonPosition));
}, Tw = function() {
  if (o(this, Rn).extraPopupElement && !o(this, Rn).editor)
    return;
  this.renderCommentButton();
  const [t, e] = o(this, Kr), {
    style: n
  } = o(this, nn);
  n.left = `calc(${t}%)`, n.top = `calc(${e}% - var(--comment-button-dim))`;
}, kw = function() {
  o(this, Rn).extraPopupElement || (this.renderCommentButton(), o(this, nn).style.backgroundColor = this.commentButtonColor || "");
}, Cm = function() {
  const t = o(this, Qr), e = o(this, tr);
  return t != null && t.str && (!(e != null && e.str) || e.str === t.str) && o(this, Qr).html || null;
}, Pw = function() {
  var t, e, n;
  return ((n = (e = (t = o(this, Nt, Cm)) == null ? void 0 : t.attributes) == null ? void 0 : e.style) == null ? void 0 : n.fontSize) || 0;
}, N_ = function() {
  var t, e, n;
  return ((n = (e = (t = o(this, Nt, Cm)) == null ? void 0 : t.attributes) == null ? void 0 : e.style) == null ? void 0 : n.color) || null;
}, O_ = function(t) {
  const e = [], n = {
    str: t,
    html: {
      name: "div",
      attributes: {
        dir: "auto"
      },
      children: [{
        name: "p",
        children: e
      }]
    }
  }, s = {
    style: {
      color: o(this, Nt, N_),
      fontSize: o(this, Nt, Pw) ? `calc(${o(this, Nt, Pw)}px * var(--total-scale-factor))` : ""
    }
  };
  for (const a of t.split(`
`))
    e.push({
      name: "span",
      value: a,
      attributes: s
    });
  return n;
}, B_ = function(t) {
  t.altKey || t.shiftKey || t.ctrlKey || t.metaKey || (t.key === "Enter" || t.key === "Escape" && o(this, er)) && E(this, Nt, xm).call(this);
}, z_ = function() {
  if (o(this, Yr) !== null)
    return;
  const {
    page: {
      view: t
    },
    viewport: {
      rawDims: {
        pageWidth: e,
        pageHeight: n,
        pageX: s,
        pageY: a
      }
    }
  } = o(this, nh);
  let l = !!o(this, ih), u = l ? o(this, ih) : o(this, op);
  for (const x of o(this, Ul))
    if (!u || ft.intersect(x.data.rect, u) !== null) {
      u = x.data.rect, l = !0;
      break;
    }
  const h = ft.normalizeRect([u[0], t[3] - u[1] + t[1], u[2], t[3] - u[3] + t[1]]), g = l ? u[2] - u[0] + 5 : 0, y = h[0] + g, v = h[1];
  w(this, Yr, [100 * (y - s) / e, 100 * (v - a) / n]);
  const {
    style: S
  } = o(this, Me);
  S.left = `${o(this, Yr)[0]}%`, S.top = `${o(this, Yr)[1]}%`;
}, xm = function() {
  if (o(this, yi)) {
    o(this, yi).toggleCommentPopup(this, !1);
    return;
  }
  w(this, er, !o(this, er)), o(this, er) ? (E(this, Nt, Tm).call(this), o(this, Me).addEventListener("click", o(this, th)), o(this, Me).addEventListener("keydown", o(this, aa))) : (E(this, Nt, Rw).call(this), o(this, Me).removeEventListener("click", o(this, th)), o(this, Me).removeEventListener("keydown", o(this, aa)));
}, Tm = function() {
  o(this, vi) || this.render(), this.isVisible ? o(this, er) && o(this, Me).classList.add("focused") : (E(this, Nt, z_).call(this), o(this, Me).hidden = !1, o(this, Me).style.zIndex = parseInt(o(this, Me).style.zIndex) + 1e3);
}, Rw = function() {
  o(this, Me).classList.remove("focused"), !(o(this, er) || !this.isVisible) && (o(this, Me).hidden = !0, o(this, Me).style.zIndex = parseInt(o(this, Me).style.zIndex) - 1e3);
};
class U_ extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0
    }), this.textContent = t.data.textContent, this.textPosition = t.data.textPosition, this.annotationEditorType = Pt.FREETEXT;
  }
  render() {
    if (this.container.classList.add("freeTextAnnotation"), this.textContent) {
      const t = document.createElement("div");
      t.classList.add("annotationTextContent"), t.setAttribute("role", "comment");
      for (const e of this.textContent) {
        const n = document.createElement("span");
        n.textContent = e, t.append(n);
      }
      this.container.append(t);
    }
    return !this.data.popupRef && this.hasPopupData && this._createPopup(), this._editOnDoubleClick(), this.container;
  }
}
var ap;
class FP extends xe {
  constructor(e) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !0
    });
    b(this, ap, null);
  }
  render() {
    this.container.classList.add("lineAnnotation");
    const {
      data: e,
      width: n,
      height: s
    } = this, a = this.svgFactory.create(n, s, !0), l = w(this, ap, this.svgFactory.createElement("svg:line"));
    return l.setAttribute("x1", e.rect[2] - e.lineCoordinates[0]), l.setAttribute("y1", e.rect[3] - e.lineCoordinates[1]), l.setAttribute("x2", e.rect[2] - e.lineCoordinates[2]), l.setAttribute("y2", e.rect[3] - e.lineCoordinates[3]), l.setAttribute("stroke-width", e.borderStyle.width || 1), l.setAttribute("stroke", "transparent"), l.setAttribute("fill", "transparent"), a.append(l), this.container.append(a), !e.popupRef && this.hasPopupData && this._createPopup(), this.container;
  }
  getElementsToTriggerPopup() {
    return o(this, ap);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
ap = new WeakMap();
var lp;
class NP extends xe {
  constructor(e) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !0
    });
    b(this, lp, null);
  }
  render() {
    this.container.classList.add("squareAnnotation");
    const {
      data: e,
      width: n,
      height: s
    } = this, a = this.svgFactory.create(n, s, !0), l = e.borderStyle.width, u = w(this, lp, this.svgFactory.createElement("svg:rect"));
    return u.setAttribute("x", l / 2), u.setAttribute("y", l / 2), u.setAttribute("width", n - l), u.setAttribute("height", s - l), u.setAttribute("stroke-width", l || 1), u.setAttribute("stroke", "transparent"), u.setAttribute("fill", "transparent"), a.append(u), this.container.append(a), !e.popupRef && this.hasPopupData && this._createPopup(), this.container;
  }
  getElementsToTriggerPopup() {
    return o(this, lp);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
lp = new WeakMap();
var cp;
class OP extends xe {
  constructor(e) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !0
    });
    b(this, cp, null);
  }
  render() {
    this.container.classList.add("circleAnnotation");
    const {
      data: e,
      width: n,
      height: s
    } = this, a = this.svgFactory.create(n, s, !0), l = e.borderStyle.width, u = w(this, cp, this.svgFactory.createElement("svg:ellipse"));
    return u.setAttribute("cx", n / 2), u.setAttribute("cy", s / 2), u.setAttribute("rx", n / 2 - l / 2), u.setAttribute("ry", s / 2 - l / 2), u.setAttribute("stroke-width", l || 1), u.setAttribute("stroke", "transparent"), u.setAttribute("fill", "transparent"), a.append(u), this.container.append(a), !e.popupRef && this.hasPopupData && this._createPopup(), this.container;
  }
  getElementsToTriggerPopup() {
    return o(this, cp);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
cp = new WeakMap();
var up;
class H_ extends xe {
  constructor(e) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !0
    });
    b(this, up, null);
    this.containerClassName = "polylineAnnotation", this.svgElementName = "svg:polyline";
  }
  render() {
    this.container.classList.add(this.containerClassName);
    const {
      data: {
        rect: e,
        vertices: n,
        borderStyle: s,
        popupRef: a
      },
      width: l,
      height: u
    } = this;
    if (!n)
      return this.container;
    const h = this.svgFactory.create(l, u, !0);
    let f = [];
    for (let y = 0, v = n.length; y < v; y += 2) {
      const S = n[y] - e[0], x = e[3] - n[y + 1];
      f.push(`${S},${x}`);
    }
    f = f.join(" ");
    const g = w(this, up, this.svgFactory.createElement(this.svgElementName));
    return g.setAttribute("points", f), g.setAttribute("stroke-width", s.width || 1), g.setAttribute("stroke", "transparent"), g.setAttribute("fill", "transparent"), h.append(g), this.container.append(h), !a && this.hasPopupData && this._createPopup(), this.container;
  }
  getElementsToTriggerPopup() {
    return o(this, up);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
up = new WeakMap();
class BP extends H_ {
  constructor(t) {
    super(t), this.containerClassName = "polygonAnnotation", this.svgElementName = "svg:polygon";
  }
}
class zP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0
    });
  }
  render() {
    return this.container.classList.add("caretAnnotation"), !this.data.popupRef && this.hasPopupData && this._createPopup(), this.container;
  }
}
var hp, jl, dp, Mw;
class AA extends xe {
  constructor(e) {
    super(e, {
      isRenderable: !0,
      ignoreBorder: !0
    });
    b(this, dp);
    b(this, hp, null);
    b(this, jl, []);
    this.containerClassName = "inkAnnotation", this.svgElementName = "svg:polyline", this.annotationEditorType = this.data.it === "InkHighlight" ? Pt.HIGHLIGHT : Pt.INK;
  }
  render() {
    this.container.classList.add(this.containerClassName);
    const {
      data: {
        rect: e,
        rotation: n,
        inkLists: s,
        borderStyle: a,
        popupRef: l
      }
    } = this, {
      transform: u,
      width: h,
      height: f
    } = E(this, dp, Mw).call(this, n, e), g = this.svgFactory.create(h, f, !0), y = w(this, hp, this.svgFactory.createElement("svg:g"));
    g.append(y), y.setAttribute("stroke-width", a.width || 1), y.setAttribute("stroke-linecap", "round"), y.setAttribute("stroke-linejoin", "round"), y.setAttribute("stroke-miterlimit", 10), y.setAttribute("stroke", "transparent"), y.setAttribute("fill", "transparent"), y.setAttribute("transform", u);
    for (let v = 0, S = s.length; v < S; v++) {
      const x = this.svgFactory.createElement(this.svgElementName);
      o(this, jl).push(x), x.setAttribute("points", s[v].join(",")), y.append(x);
    }
    return !l && this.hasPopupData && this._createPopup(), this.container.append(g), this._editOnDoubleClick(), this.container;
  }
  updateEdited(e) {
    super.updateEdited(e);
    const {
      thickness: n,
      points: s,
      rect: a
    } = e, l = o(this, hp);
    if (n >= 0 && l.setAttribute("stroke-width", n || 1), s)
      for (let u = 0, h = o(this, jl).length; u < h; u++)
        o(this, jl)[u].setAttribute("points", s[u].join(","));
    if (a) {
      const {
        transform: u,
        width: h,
        height: f
      } = E(this, dp, Mw).call(this, this.data.rotation, a);
      l.parentElement.setAttribute("viewBox", `0 0 ${h} ${f}`), l.setAttribute("transform", u);
    }
  }
  getElementsToTriggerPopup() {
    return o(this, jl);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
hp = new WeakMap(), jl = new WeakMap(), dp = new WeakSet(), Mw = function(e, n) {
  switch (e) {
    case 90:
      return {
        transform: `rotate(90) translate(${-n[0]},${n[1]}) scale(1,-1)`,
        width: n[3] - n[1],
        height: n[2] - n[0]
      };
    case 180:
      return {
        transform: `rotate(180) translate(${-n[2]},${n[1]}) scale(1,-1)`,
        width: n[2] - n[0],
        height: n[3] - n[1]
      };
    case 270:
      return {
        transform: `rotate(270) translate(${-n[2]},${n[3]}) scale(1,-1)`,
        width: n[3] - n[1],
        height: n[2] - n[0]
      };
    default:
      return {
        transform: `translate(${-n[0]},${n[3]}) scale(1,-1)`,
        width: n[2] - n[0],
        height: n[3] - n[1]
      };
  }
};
class j_ extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0,
      createQuadrilaterals: !0
    }), this.annotationEditorType = Pt.HIGHLIGHT;
  }
  render() {
    const {
      data: {
        overlaidText: t,
        popupRef: e
      }
    } = this;
    if (!e && this.hasPopupData && this._createPopup(), this.container.classList.add("highlightAnnotation"), this._editOnDoubleClick(), t) {
      const n = document.createElement("mark");
      n.classList.add("overlaidText"), n.textContent = t, this.container.append(n);
    }
    return this.container;
  }
}
class UP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0,
      createQuadrilaterals: !0
    });
  }
  render() {
    const {
      data: {
        overlaidText: t,
        popupRef: e
      }
    } = this;
    if (!e && this.hasPopupData && this._createPopup(), this.container.classList.add("underlineAnnotation"), t) {
      const n = document.createElement("u");
      n.classList.add("overlaidText"), n.textContent = t, this.container.append(n);
    }
    return this.container;
  }
}
class HP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0,
      createQuadrilaterals: !0
    });
  }
  render() {
    const {
      data: {
        overlaidText: t,
        popupRef: e
      }
    } = this;
    if (!e && this.hasPopupData && this._createPopup(), this.container.classList.add("squigglyAnnotation"), t) {
      const n = document.createElement("u");
      n.classList.add("overlaidText"), n.textContent = t, this.container.append(n);
    }
    return this.container;
  }
}
class jP extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0,
      createQuadrilaterals: !0
    });
  }
  render() {
    const {
      data: {
        overlaidText: t,
        popupRef: e
      }
    } = this;
    if (!e && this.hasPopupData && this._createPopup(), this.container.classList.add("strikeoutAnnotation"), t) {
      const n = document.createElement("s");
      n.classList.add("overlaidText"), n.textContent = t, this.container.append(n);
    }
    return this.container;
  }
}
class $_ extends xe {
  constructor(t) {
    super(t, {
      isRenderable: !0,
      ignoreBorder: !0
    }), this.annotationEditorType = Pt.STAMP;
  }
  render() {
    return this.container.classList.add("stampAnnotation"), this.container.setAttribute("role", "img"), !this.data.popupRef && this.hasPopupData && this._createPopup(), this._editOnDoubleClick(), this.container;
  }
}
var fp, pp, Lw;
class $P extends xe {
  constructor(e) {
    var s;
    super(e, {
      isRenderable: !0
    });
    b(this, pp);
    b(this, fp, null);
    const {
      file: n
    } = this.data;
    this.filename = n.filename, this.content = n.content, (s = this.linkService.eventBus) == null || s.dispatch("fileattachmentannotation", {
      source: this,
      ...n
    });
  }
  render() {
    this.container.classList.add("fileAttachmentAnnotation");
    const {
      container: e,
      data: n
    } = this;
    let s;
    n.hasAppearance || n.fillAlpha === 0 ? s = document.createElement("div") : (s = document.createElement("img"), s.src = `${this.imageResourcesPath}annotation-${/paperclip/i.test(n.name) ? "paperclip" : "pushpin"}.svg`, n.fillAlpha && n.fillAlpha < 1 && (s.style = `filter: opacity(${Math.round(n.fillAlpha * 100)}%);`)), s.addEventListener("dblclick", E(this, pp, Lw).bind(this)), w(this, fp, s);
    const {
      isMac: a
    } = cn.platform;
    return e.addEventListener("keydown", (l) => {
      l.key === "Enter" && (a ? l.metaKey : l.ctrlKey) && E(this, pp, Lw).call(this);
    }), !n.popupRef && this.hasPopupData ? this._createPopup() : s.classList.add("popupTriggerArea"), e.append(s), e;
  }
  getElementsToTriggerPopup() {
    return o(this, fp);
  }
  addHighlightArea() {
    this.container.classList.add("highlightArea");
  }
}
fp = new WeakMap(), pp = new WeakSet(), Lw = function() {
  var e;
  (e = this.downloadManager) == null || e.openOrDownloadData(this.content, this.filename);
};
var rh, $l, oh, da, gp, Vl, Ra, Dw, Iw, ah;
let SA = (ah = class {
  constructor({
    div: t,
    accessibilityManager: e,
    annotationCanvasMap: n,
    annotationEditorUIManager: s,
    page: a,
    viewport: l,
    structTreeLayer: u,
    commentManager: h,
    linkService: f,
    annotationStorage: g
  }) {
    b(this, Ra);
    b(this, rh, null);
    b(this, $l, null);
    b(this, oh, null);
    b(this, da, /* @__PURE__ */ new Map());
    b(this, gp, null);
    b(this, Vl, null);
    this.div = t, w(this, rh, e), w(this, $l, n), w(this, gp, u || null), w(this, Vl, f || null), w(this, oh, g || new pA()), this.page = a, this.viewport = l, this.zIndex = 0, this._annotationEditorUIManager = s, this._commentManager = h || null;
  }
  hasEditableAnnotations() {
    return o(this, da).size > 0;
  }
  async render(t) {
    var l, u, h;
    const {
      annotations: e
    } = t, n = this.div;
    Ta(n, this.viewport);
    const s = /* @__PURE__ */ new Map(), a = {
      data: null,
      layer: n,
      linkService: o(this, Vl),
      downloadManager: t.downloadManager,
      imageResourcesPath: t.imageResourcesPath || "",
      renderForms: t.renderForms !== !1,
      svgFactory: new Zd(),
      annotationStorage: o(this, oh),
      enableComment: t.enableComment === !0,
      enableScripting: t.enableScripting === !0,
      hasJSActions: t.hasJSActions,
      fieldObjects: t.fieldObjects,
      parent: this,
      elements: null
    };
    for (const f of e) {
      if (f.noHTML)
        continue;
      const g = f.annotationType === Oe.POPUP;
      if (g) {
        const S = s.get(f.id);
        if (!S)
          continue;
        a.elements = S;
      } else if (f.rect[2] === f.rect[0] || f.rect[3] === f.rect[1])
        continue;
      a.data = f;
      const y = e1.create(a);
      if (!y.isRenderable)
        continue;
      if (!g && f.popupRef) {
        const S = s.get(f.popupRef);
        S ? S.push(y) : s.set(f.popupRef, [y]);
      }
      const v = y.render();
      f.hidden && (v.style.visibility = "hidden"), await E(this, Ra, Dw).call(this, v, f.id, a.elements), (u = (l = y.extraPopupElement) == null ? void 0 : l.popup) == null || u.renderCommentButton(), y._isEditable && (o(this, da).set(y.data.id, y), (h = this._annotationEditorUIManager) == null || h.renderAnnotationElement(y));
    }
    E(this, Ra, Iw).call(this);
  }
  async addLinkAnnotations(t) {
    const e = {
      data: null,
      layer: this.div,
      linkService: o(this, Vl),
      svgFactory: new Zd(),
      parent: this
    };
    for (const n of t) {
      n.borderStyle || (n.borderStyle = ah._defaultBorderStyle), e.data = n;
      const s = e1.create(e);
      if (!s.isRenderable)
        continue;
      const a = s.render();
      await E(this, Ra, Dw).call(this, a, n.id, null);
    }
  }
  update({
    viewport: t
  }) {
    const e = this.div;
    this.viewport = t, Ta(e, {
      rotation: t.rotation
    }), E(this, Ra, Iw).call(this), e.hidden = !1;
  }
  getEditableAnnotations() {
    return Array.from(o(this, da).values());
  }
  getEditableAnnotation(t) {
    return o(this, da).get(t);
  }
  addFakeAnnotation(t) {
    var u;
    const {
      div: e
    } = this, {
      id: n,
      rotation: s
    } = t, a = new TP({
      data: {
        id: n,
        rect: t.getPDFRect(),
        rotation: s
      },
      editor: t,
      layer: e,
      parent: this,
      enableComment: !!this._commentManager,
      linkService: o(this, Vl),
      annotationStorage: o(this, oh)
    }), l = a.render();
    return e.append(l), (u = o(this, rh)) == null || u.moveElementInDOM(e, l, l, !1), a.createOrUpdatePopup(), a;
  }
  static get _defaultBorderStyle() {
    return It(this, "_defaultBorderStyle", Object.freeze({
      width: 1,
      rawWidth: 1,
      style: Hc.SOLID,
      dashArray: [3],
      horizontalCornerRadius: 0,
      verticalCornerRadius: 0
    }));
  }
}, rh = new WeakMap(), $l = new WeakMap(), oh = new WeakMap(), da = new WeakMap(), gp = new WeakMap(), Vl = new WeakMap(), Ra = new WeakSet(), Dw = async function(t, e, n) {
  var u, h;
  const s = t.firstChild || t, a = s.id = `${hA}${e}`, l = await ((u = o(this, gp)) == null ? void 0 : u.getAriaAttributes(a));
  if (l)
    for (const [f, g] of l)
      s.setAttribute(f, g);
  n ? n.at(-1).container.after(t) : (this.div.append(t), (h = o(this, rh)) == null || h.moveElementInDOM(this.div, t, s, !1));
}, Iw = function() {
  var e;
  if (!o(this, $l))
    return;
  const t = this.div;
  for (const [n, s] of o(this, $l)) {
    const a = t.querySelector(`[data-annotation-id="${n}"]`);
    if (!a)
      continue;
    s.className = "annotationContent";
    const {
      firstChild: l
    } = a;
    l ? l.nodeName === "CANVAS" ? l.replaceWith(s) : l.classList.contains("annotationContent") ? l.after(s) : l.before(s) : a.append(s);
    const u = o(this, da).get(n);
    u && (u._hasNoCanvas ? ((e = this._annotationEditorUIManager) == null || e.setMissingCanvas(n, a.id, s), u._hasNoCanvas = !1) : u.canvas = s);
  }
  o(this, $l).clear();
}, ah);
const lm = /\r\n?|\n/g;
var wi, mp, Wl, Ai, $e, V_, W_, G_, km, oo, Pm, Rm, X_, Nw, q_;
const me = class me extends fe {
  constructor(e) {
    super({
      ...e,
      name: "freeTextEditor"
    });
    b(this, $e);
    b(this, wi, "");
    b(this, mp, `${this.id}-editor`);
    b(this, Wl, null);
    b(this, Ai);
    Z(this, "_colorPicker", null);
    this.color = e.color || me._defaultColor || fe._defaultLineColor, w(this, Ai, e.fontSize || me._defaultFontSize), this.annotationElementId || this._uiManager.a11yAlert("pdfjs-editor-freetext-added-alert");
  }
  static get _keyboardManager() {
    const e = me.prototype, n = (l) => l.isEmpty(), s = ka.TRANSLATE_SMALL, a = ka.TRANSLATE_BIG;
    return It(this, "_keyboardManager", new Bp([[["ctrl+s", "mac+meta+s", "ctrl+p", "mac+meta+p"], e.commitOrRemove, {
      bubbles: !0
    }], [["ctrl+Enter", "mac+meta+Enter", "Escape", "mac+Escape"], e.commitOrRemove], [["ArrowLeft", "mac+ArrowLeft"], e._translateEmpty, {
      args: [-s, 0],
      checker: n
    }], [["ctrl+ArrowLeft", "mac+shift+ArrowLeft"], e._translateEmpty, {
      args: [-a, 0],
      checker: n
    }], [["ArrowRight", "mac+ArrowRight"], e._translateEmpty, {
      args: [s, 0],
      checker: n
    }], [["ctrl+ArrowRight", "mac+shift+ArrowRight"], e._translateEmpty, {
      args: [a, 0],
      checker: n
    }], [["ArrowUp", "mac+ArrowUp"], e._translateEmpty, {
      args: [0, -s],
      checker: n
    }], [["ctrl+ArrowUp", "mac+shift+ArrowUp"], e._translateEmpty, {
      args: [0, -a],
      checker: n
    }], [["ArrowDown", "mac+ArrowDown"], e._translateEmpty, {
      args: [0, s],
      checker: n
    }], [["ctrl+ArrowDown", "mac+shift+ArrowDown"], e._translateEmpty, {
      args: [0, a],
      checker: n
    }]]));
  }
  static initialize(e, n) {
    fe.initialize(e, n);
    const s = getComputedStyle(document.documentElement);
    this._internalPadding = parseFloat(s.getPropertyValue("--freetext-padding"));
  }
  static updateDefaultParams(e, n) {
    switch (e) {
      case jt.FREETEXT_SIZE:
        me._defaultFontSize = n;
        break;
      case jt.FREETEXT_COLOR:
        me._defaultColor = n;
        break;
    }
  }
  updateParams(e, n) {
    switch (e) {
      case jt.FREETEXT_SIZE:
        E(this, $e, V_).call(this, n);
        break;
      case jt.FREETEXT_COLOR:
        E(this, $e, W_).call(this, n);
        break;
    }
  }
  static get defaultPropertiesToUpdate() {
    return [[jt.FREETEXT_SIZE, me._defaultFontSize], [jt.FREETEXT_COLOR, me._defaultColor || fe._defaultLineColor]];
  }
  get propertiesToUpdate() {
    return [[jt.FREETEXT_SIZE, o(this, Ai)], [jt.FREETEXT_COLOR, this.color]];
  }
  get toolbarButtons() {
    return this._colorPicker || (this._colorPicker = new Ym(this)), [["colorPicker", this._colorPicker]];
  }
  get colorType() {
    return jt.FREETEXT_COLOR;
  }
  onUpdatedColor() {
    var e;
    this.editorDiv.style.color = this.color, (e = this._colorPicker) == null || e.update(this.color), super.onUpdatedColor();
  }
  _translateEmpty(e, n) {
    this._uiManager.translateSelectedEditors(e, n, !0);
  }
  getInitialTranslation() {
    const e = this.parentScale;
    return [-me._internalPadding * e, -(me._internalPadding + o(this, Ai)) * e];
  }
  rebuild() {
    this.parent && (super.rebuild(), this.div !== null && (this.isAttachedToDOM || this.parent.add(this)));
  }
  enableEditMode() {
    if (!super.enableEditMode())
      return !1;
    this.overlayDiv.classList.remove("enabled"), this.editorDiv.contentEditable = !0, this._isDraggable = !1, this.div.removeAttribute("aria-activedescendant"), w(this, Wl, new AbortController());
    const e = this._uiManager.combinedSignal(o(this, Wl));
    return this.editorDiv.addEventListener("keydown", this.editorDivKeydown.bind(this), {
      signal: e
    }), this.editorDiv.addEventListener("focus", this.editorDivFocus.bind(this), {
      signal: e
    }), this.editorDiv.addEventListener("blur", this.editorDivBlur.bind(this), {
      signal: e
    }), this.editorDiv.addEventListener("input", this.editorDivInput.bind(this), {
      signal: e
    }), this.editorDiv.addEventListener("paste", this.editorDivPaste.bind(this), {
      signal: e
    }), !0;
  }
  disableEditMode() {
    var e;
    return super.disableEditMode() ? (this.overlayDiv.classList.add("enabled"), this.editorDiv.contentEditable = !1, this.div.setAttribute("aria-activedescendant", o(this, mp)), this._isDraggable = !0, (e = o(this, Wl)) == null || e.abort(), w(this, Wl, null), this.div.focus({
      preventScroll: !0
    }), this.isEditing = !1, this.parent.div.classList.add("freetextEditing"), !0) : !1;
  }
  focusin(e) {
    this._focusEventsAllowed && (super.focusin(e), e.target !== this.editorDiv && this.editorDiv.focus());
  }
  onceAdded(e) {
    var n;
    this.width || (this.enableEditMode(), e && this.editorDiv.focus(), (n = this._initialOptions) != null && n.isCentered && this.center(), this._initialOptions = null);
  }
  isEmpty() {
    return !this.editorDiv || this.editorDiv.innerText.trim() === "";
  }
  remove() {
    this.isEditing = !1, this.parent && (this.parent.setEditingState(!0), this.parent.div.classList.add("freetextEditing")), super.remove();
  }
  commit() {
    if (!this.isInEditMode())
      return;
    super.commit(), this.disableEditMode();
    const e = o(this, wi), n = w(this, wi, E(this, $e, G_).call(this).trimEnd());
    if (e === n)
      return;
    const s = (a) => {
      if (w(this, wi, a), !a) {
        this.remove();
        return;
      }
      E(this, $e, Rm).call(this), this._uiManager.rebuild(this), E(this, $e, km).call(this);
    };
    this.addCommands({
      cmd: () => {
        s(n);
      },
      undo: () => {
        s(e);
      },
      mustExec: !1
    }), E(this, $e, km).call(this);
  }
  shouldGetKeyboardEvents() {
    return this.isInEditMode();
  }
  enterInEditMode() {
    this.enableEditMode(), this.editorDiv.focus();
  }
  keydown(e) {
    e.target === this.div && e.key === "Enter" && (this.enterInEditMode(), e.preventDefault());
  }
  editorDivKeydown(e) {
    me._keyboardManager.exec(this, e);
  }
  editorDivFocus(e) {
    this.isEditing = !0;
  }
  editorDivBlur(e) {
    this.isEditing = !1;
  }
  editorDivInput(e) {
    this.parent.div.classList.toggle("freetextEditing", this.isEmpty());
  }
  disableEditing() {
    this.editorDiv.setAttribute("role", "comment"), this.editorDiv.removeAttribute("aria-multiline");
  }
  enableEditing() {
    this.editorDiv.setAttribute("role", "textbox"), this.editorDiv.setAttribute("aria-multiline", !0);
  }
  get canChangeContent() {
    return !0;
  }
  render() {
    if (this.div)
      return this.div;
    let e, n;
    (this._isCopy || this.annotationElementId) && (e = this.x, n = this.y), super.render(), this.editorDiv = document.createElement("div"), this.editorDiv.className = "internal", this.editorDiv.setAttribute("id", o(this, mp)), this.editorDiv.setAttribute("data-l10n-id", "pdfjs-free-text2"), this.editorDiv.setAttribute("data-l10n-attrs", "default-content"), this.enableEditing(), this.editorDiv.contentEditable = !0;
    const {
      style: s
    } = this.editorDiv;
    if (s.fontSize = `calc(${o(this, Ai)}px * var(--total-scale-factor))`, s.color = this.color, this.div.append(this.editorDiv), this.overlayDiv = document.createElement("div"), this.overlayDiv.classList.add("overlay", "enabled"), this.div.append(this.overlayDiv), this._isCopy || this.annotationElementId) {
      const [a, l] = this.parentDimensions;
      if (this.annotationElementId) {
        const {
          position: u
        } = this._initialData;
        let [h, f] = this.getInitialTranslation();
        [h, f] = this.pageTranslationToScreen(h, f);
        const [g, y] = this.pageDimensions, [v, S] = this.pageTranslation;
        let x, _;
        switch (this.rotation) {
          case 0:
            x = e + (u[0] - v) / g, _ = n + this.height - (u[1] - S) / y;
            break;
          case 90:
            x = e + (u[0] - v) / g, _ = n - (u[1] - S) / y, [h, f] = [f, -h];
            break;
          case 180:
            x = e - this.width + (u[0] - v) / g, _ = n - (u[1] - S) / y, [h, f] = [-h, -f];
            break;
          case 270:
            x = e + (u[0] - v - this.height * y) / g, _ = n + (u[1] - S - this.width * g) / y, [h, f] = [-f, h];
            break;
        }
        this.setAt(x * a, _ * l, h, f);
      } else
        this._moveAfterPaste(e, n);
      E(this, $e, Rm).call(this), this._isDraggable = !0, this.editorDiv.contentEditable = !1;
    } else
      this._isDraggable = !1, this.editorDiv.contentEditable = !0;
    return this.div;
  }
  editorDivPaste(e) {
    var x, _, k;
    const n = e.clipboardData || window.clipboardData, {
      types: s
    } = n;
    if (s.length === 1 && s[0] === "text/plain")
      return;
    e.preventDefault();
    const a = E(x = me, oo, Nw).call(x, n.getData("text") || "").replaceAll(lm, `
`);
    if (!a)
      return;
    const l = window.getSelection();
    if (!l.rangeCount)
      return;
    this.editorDiv.normalize(), l.deleteFromDocument();
    const u = l.getRangeAt(0);
    if (!a.includes(`
`)) {
      u.insertNode(document.createTextNode(a)), this.editorDiv.normalize(), l.collapseToStart();
      return;
    }
    const {
      startContainer: h,
      startOffset: f
    } = u, g = [], y = [];
    if (h.nodeType === Node.TEXT_NODE) {
      const T = h.parentElement;
      if (y.push(h.nodeValue.slice(f).replaceAll(lm, "")), T !== this.editorDiv) {
        let R = g;
        for (const P of this.editorDiv.childNodes) {
          if (P === T) {
            R = y;
            continue;
          }
          R.push(E(_ = me, oo, Pm).call(_, P));
        }
      }
      g.push(h.nodeValue.slice(0, f).replaceAll(lm, ""));
    } else if (h === this.editorDiv) {
      let T = g, R = 0;
      for (const P of this.editorDiv.childNodes)
        R++ === f && (T = y), T.push(E(k = me, oo, Pm).call(k, P));
    }
    w(this, wi, `${g.join(`
`)}${a}${y.join(`
`)}`), E(this, $e, Rm).call(this);
    const v = new Range();
    let S = Math.sumPrecise(g.map((T) => T.length));
    for (const {
      firstChild: T
    } of this.editorDiv.childNodes)
      if (T.nodeType === Node.TEXT_NODE) {
        const R = T.nodeValue.length;
        if (S <= R) {
          v.setStart(T, S), v.setEnd(T, S);
          break;
        }
        S -= R;
      }
    l.removeAllRanges(), l.addRange(v);
  }
  get contentDiv() {
    return this.editorDiv;
  }
  getPDFRect() {
    const e = me._internalPadding * this.parentScale;
    return this.getRect(e, e);
  }
  static async deserialize(e, n, s) {
    var u;
    let a = null;
    if (e instanceof U_) {
      const {
        data: {
          defaultAppearanceData: {
            fontSize: h,
            fontColor: f
          },
          rect: g,
          rotation: y,
          id: v,
          popupRef: S,
          richText: x,
          contentsObj: _,
          creationDate: k,
          modificationDate: T
        },
        textContent: R,
        textPosition: P,
        parent: {
          page: {
            pageNumber: L
          }
        }
      } = e;
      if (!R || R.length === 0)
        return null;
      a = e = {
        annotationType: Pt.FREETEXT,
        color: Array.from(f),
        fontSize: h,
        value: R.join(`
`),
        position: P,
        pageIndex: L - 1,
        rect: g.slice(0),
        rotation: y,
        annotationElementId: v,
        id: v,
        deleted: !1,
        popupRef: S,
        comment: (_ == null ? void 0 : _.str) || null,
        richText: x,
        creationDate: k,
        modificationDate: T
      };
    }
    const l = await super.deserialize(e, n, s);
    return w(l, Ai, e.fontSize), l.color = ft.makeHexColor(...e.color), w(l, wi, E(u = me, oo, Nw).call(u, e.value)), l._initialData = a, e.comment && l.setCommentData(e), l;
  }
  serialize(e = !1) {
    if (this.isEmpty())
      return null;
    if (this.deleted)
      return this.serializeDeleted();
    const n = fe._colorManager.convert(this.isAttachedToDOM ? getComputedStyle(this.editorDiv).color : this.color), s = Object.assign(super.serialize(e), {
      color: n,
      fontSize: o(this, Ai),
      value: E(this, $e, X_).call(this)
    });
    return this.addComment(s), e ? (s.isCopy = !0, s) : this.annotationElementId && !E(this, $e, q_).call(this, s) ? null : (s.id = this.annotationElementId, s);
  }
  renderAnnotationElement(e) {
    const n = super.renderAnnotationElement(e);
    if (!n)
      return null;
    const {
      style: s
    } = n;
    s.fontSize = `calc(${o(this, Ai)}px * var(--total-scale-factor))`, s.color = this.color, n.replaceChildren();
    for (const a of o(this, wi).split(`
`)) {
      const l = document.createElement("div");
      l.append(a ? document.createTextNode(a) : document.createElement("br")), n.append(l);
    }
    return e.updateEdited({
      rect: this.getPDFRect(),
      popup: this._uiManager.hasCommentManager() || this.hasEditedComment ? this.comment : {
        text: o(this, wi)
      }
    }), n;
  }
  resetAnnotationElement(e) {
    super.resetAnnotationElement(e), e.resetEdited();
  }
};
wi = new WeakMap(), mp = new WeakMap(), Wl = new WeakMap(), Ai = new WeakMap(), $e = new WeakSet(), V_ = function(e) {
  const n = (a) => {
    this.editorDiv.style.fontSize = `calc(${a}px * var(--total-scale-factor))`, this.translate(0, -(a - o(this, Ai)) * this.parentScale), w(this, Ai, a), E(this, $e, km).call(this);
  }, s = o(this, Ai);
  this.addCommands({
    cmd: n.bind(this, e),
    undo: n.bind(this, s),
    post: this._uiManager.updateUI.bind(this._uiManager, this),
    mustExec: !0,
    type: jt.FREETEXT_SIZE,
    overwriteIfSameType: !0,
    keepUndo: !0
  });
}, W_ = function(e) {
  const n = (a) => {
    this.color = a, this.onUpdatedColor();
  }, s = this.color;
  this.addCommands({
    cmd: n.bind(this, e),
    undo: n.bind(this, s),
    post: this._uiManager.updateUI.bind(this._uiManager, this),
    mustExec: !0,
    type: jt.FREETEXT_COLOR,
    overwriteIfSameType: !0,
    keepUndo: !0
  });
}, G_ = function() {
  var s;
  const e = [];
  this.editorDiv.normalize();
  let n = null;
  for (const a of this.editorDiv.childNodes)
    (n == null ? void 0 : n.nodeType) === Node.TEXT_NODE && a.nodeName === "BR" || (e.push(E(s = me, oo, Pm).call(s, a)), n = a);
  return e.join(`
`);
}, km = function() {
  const [e, n] = this.parentDimensions;
  let s;
  if (this.isAttachedToDOM)
    s = this.div.getBoundingClientRect();
  else {
    const {
      currentLayer: a,
      div: l
    } = this, u = l.style.display, h = l.classList.contains("hidden");
    l.classList.remove("hidden"), l.style.display = "hidden", a.div.append(this.div), s = l.getBoundingClientRect(), l.remove(), l.style.display = u, l.classList.toggle("hidden", h);
  }
  this.rotation % 180 === this.parentRotation % 180 ? (this.width = s.width / e, this.height = s.height / n) : (this.width = s.height / e, this.height = s.width / n), this.fixAndSetPosition();
}, oo = new WeakSet(), Pm = function(e) {
  return (e.nodeType === Node.TEXT_NODE ? e.nodeValue : e.innerText).replaceAll(lm, "");
}, Rm = function() {
  if (this.editorDiv.replaceChildren(), !!o(this, wi))
    for (const e of o(this, wi).split(`
`)) {
      const n = document.createElement("div");
      n.append(e ? document.createTextNode(e) : document.createElement("br")), this.editorDiv.append(n);
    }
}, X_ = function() {
  return o(this, wi).replaceAll(" ", " ");
}, Nw = function(e) {
  return e.replaceAll(" ", " ");
}, q_ = function(e) {
  const {
    value: n,
    fontSize: s,
    color: a,
    pageIndex: l
  } = this._initialData;
  return this.hasEditedComment || this._hasBeenMoved || e.value !== n || e.fontSize !== s || e.color.some((u, h) => u !== a[h]) || e.pageIndex !== l;
}, b(me, oo), Z(me, "_freeTextDefaultContent", ""), Z(me, "_internalPadding", 0), Z(me, "_defaultColor", null), Z(me, "_defaultFontSize", 10), Z(me, "_type", "freetext"), Z(me, "_editorType", Pt.FREETEXT);
let Fw = me;
class dt {
  toSVGPath() {
    ce("Abstract method `toSVGPath` must be implemented.");
  }
  get box() {
    ce("Abstract getter `box` must be implemented.");
  }
  serialize(t, e) {
    ce("Abstract method `serialize` must be implemented.");
  }
  static _rescale(t, e, n, s, a, l) {
    l || (l = new Float32Array(t.length));
    for (let u = 0, h = t.length; u < h; u += 2)
      l[u] = e + t[u] * s, l[u + 1] = n + t[u + 1] * a;
    return l;
  }
  static _rescaleAndSwap(t, e, n, s, a, l) {
    l || (l = new Float32Array(t.length));
    for (let u = 0, h = t.length; u < h; u += 2)
      l[u] = e + t[u + 1] * s, l[u + 1] = n + t[u] * a;
    return l;
  }
  static _translate(t, e, n, s) {
    s || (s = new Float32Array(t.length));
    for (let a = 0, l = t.length; a < l; a += 2)
      s[a] = e + t[a], s[a + 1] = n + t[a + 1];
    return s;
  }
  static svgRound(t) {
    return Math.round(t * 1e4);
  }
  static _normalizePoint(t, e, n, s, a) {
    switch (a) {
      case 90:
        return [1 - e / n, t / s];
      case 180:
        return [1 - t / n, 1 - e / s];
      case 270:
        return [e / n, 1 - t / s];
      default:
        return [t / n, e / s];
    }
  }
  static _normalizePagePoint(t, e, n) {
    switch (n) {
      case 90:
        return [1 - e, t];
      case 180:
        return [1 - t, 1 - e];
      case 270:
        return [e, 1 - t];
      default:
        return [t, e];
    }
  }
  static createBezierPoints(t, e, n, s, a, l) {
    return [(t + 5 * n) / 6, (e + 5 * s) / 6, (5 * n + a) / 6, (5 * s + l) / 6, (n + a) / 2, (s + l) / 2];
  }
}
Z(dt, "PRECISION", 1e-4);
var Si, bs, lh, ch, nr, Ut, Gl, Xl, yp, vp, uh, hh, fa, wp, gy, my, Ke, zd, Y_, K_, Q_, Z_, J_, tC;
const _r = class _r {
  constructor({
    x: t,
    y: e
  }, n, s, a, l, u = 0) {
    b(this, Ke);
    b(this, Si);
    b(this, bs, []);
    b(this, lh);
    b(this, ch);
    b(this, nr, []);
    b(this, Ut, new Float32Array(18));
    b(this, Gl);
    b(this, Xl);
    b(this, yp);
    b(this, vp);
    b(this, uh);
    b(this, hh);
    b(this, fa, []);
    w(this, Si, n), w(this, hh, a * s), w(this, ch, l), o(this, Ut).set([NaN, NaN, NaN, NaN, t, e], 6), w(this, lh, u), w(this, vp, o(_r, wp) * s), w(this, yp, o(_r, my) * s), w(this, uh, s), o(this, fa).push(t, e);
  }
  isEmpty() {
    return isNaN(o(this, Ut)[8]);
  }
  add({
    x: t,
    y: e
  }) {
    var V;
    w(this, Gl, t), w(this, Xl, e);
    const [n, s, a, l] = o(this, Si);
    let [u, h, f, g] = o(this, Ut).subarray(8, 12);
    const y = t - f, v = e - g, S = Math.hypot(y, v);
    if (S < o(this, yp))
      return !1;
    const x = S - o(this, vp), _ = x / S, k = _ * y, T = _ * v;
    let R = u, P = h;
    u = f, h = g, f += k, g += T, (V = o(this, fa)) == null || V.push(t, e);
    const L = -T / x, D = k / x, N = L * o(this, hh), B = D * o(this, hh);
    return o(this, Ut).set(o(this, Ut).subarray(2, 8), 0), o(this, Ut).set([f + N, g + B], 4), o(this, Ut).set(o(this, Ut).subarray(14, 18), 12), o(this, Ut).set([f - N, g - B], 16), isNaN(o(this, Ut)[6]) ? (o(this, nr).length === 0 && (o(this, Ut).set([u + N, h + B], 2), o(this, nr).push(NaN, NaN, NaN, NaN, (u + N - n) / a, (h + B - s) / l), o(this, Ut).set([u - N, h - B], 14), o(this, bs).push(NaN, NaN, NaN, NaN, (u - N - n) / a, (h - B - s) / l)), o(this, Ut).set([R, P, u, h, f, g], 6), !this.isEmpty()) : (o(this, Ut).set([R, P, u, h, f, g], 6), Math.abs(Math.atan2(P - h, R - u) - Math.atan2(T, k)) < Math.PI / 2 ? ([u, h, f, g] = o(this, Ut).subarray(2, 6), o(this, nr).push(NaN, NaN, NaN, NaN, ((u + f) / 2 - n) / a, ((h + g) / 2 - s) / l), [u, h, R, P] = o(this, Ut).subarray(14, 18), o(this, bs).push(NaN, NaN, NaN, NaN, ((R + u) / 2 - n) / a, ((P + h) / 2 - s) / l), !0) : ([R, P, u, h, f, g] = o(this, Ut).subarray(0, 6), o(this, nr).push(((R + 5 * u) / 6 - n) / a, ((P + 5 * h) / 6 - s) / l, ((5 * u + f) / 6 - n) / a, ((5 * h + g) / 6 - s) / l, ((u + f) / 2 - n) / a, ((h + g) / 2 - s) / l), [f, g, u, h, R, P] = o(this, Ut).subarray(12, 18), o(this, bs).push(((R + 5 * u) / 6 - n) / a, ((P + 5 * h) / 6 - s) / l, ((5 * u + f) / 6 - n) / a, ((5 * h + g) / 6 - s) / l, ((u + f) / 2 - n) / a, ((h + g) / 2 - s) / l), !0));
  }
  toSVGPath() {
    if (this.isEmpty())
      return "";
    const t = o(this, nr), e = o(this, bs);
    if (isNaN(o(this, Ut)[6]) && !this.isEmpty())
      return E(this, Ke, Y_).call(this);
    const n = [];
    n.push(`M${t[4]} ${t[5]}`);
    for (let s = 6; s < t.length; s += 6)
      isNaN(t[s]) ? n.push(`L${t[s + 4]} ${t[s + 5]}`) : n.push(`C${t[s]} ${t[s + 1]} ${t[s + 2]} ${t[s + 3]} ${t[s + 4]} ${t[s + 5]}`);
    E(this, Ke, Q_).call(this, n);
    for (let s = e.length - 6; s >= 6; s -= 6)
      isNaN(e[s]) ? n.push(`L${e[s + 4]} ${e[s + 5]}`) : n.push(`C${e[s]} ${e[s + 1]} ${e[s + 2]} ${e[s + 3]} ${e[s + 4]} ${e[s + 5]}`);
    return E(this, Ke, K_).call(this, n), n.join(" ");
  }
  newFreeDrawOutline(t, e, n, s, a, l) {
    return new eC(t, e, n, s, a, l);
  }
  getOutlines() {
    var y;
    const t = o(this, nr), e = o(this, bs), n = o(this, Ut), [s, a, l, u] = o(this, Si), h = new Float32Array((((y = o(this, fa)) == null ? void 0 : y.length) ?? 0) + 2);
    for (let v = 0, S = h.length - 2; v < S; v += 2)
      h[v] = (o(this, fa)[v] - s) / l, h[v + 1] = (o(this, fa)[v + 1] - a) / u;
    if (h[h.length - 2] = (o(this, Gl) - s) / l, h[h.length - 1] = (o(this, Xl) - a) / u, isNaN(n[6]) && !this.isEmpty())
      return E(this, Ke, Z_).call(this, h);
    const f = new Float32Array(o(this, nr).length + 24 + o(this, bs).length);
    let g = t.length;
    for (let v = 0; v < g; v += 2) {
      if (isNaN(t[v])) {
        f[v] = f[v + 1] = NaN;
        continue;
      }
      f[v] = t[v], f[v + 1] = t[v + 1];
    }
    g = E(this, Ke, tC).call(this, f, g);
    for (let v = e.length - 6; v >= 6; v -= 6)
      for (let S = 0; S < 6; S += 2) {
        if (isNaN(e[v + S])) {
          f[g] = f[g + 1] = NaN, g += 2;
          continue;
        }
        f[g] = e[v + S], f[g + 1] = e[v + S + 1], g += 2;
      }
    return E(this, Ke, J_).call(this, f, g), this.newFreeDrawOutline(f, h, o(this, Si), o(this, uh), o(this, lh), o(this, ch));
  }
};
Si = new WeakMap(), bs = new WeakMap(), lh = new WeakMap(), ch = new WeakMap(), nr = new WeakMap(), Ut = new WeakMap(), Gl = new WeakMap(), Xl = new WeakMap(), yp = new WeakMap(), vp = new WeakMap(), uh = new WeakMap(), hh = new WeakMap(), fa = new WeakMap(), wp = new WeakMap(), gy = new WeakMap(), my = new WeakMap(), Ke = new WeakSet(), zd = function() {
  const t = o(this, Ut).subarray(4, 6), e = o(this, Ut).subarray(16, 18), [n, s, a, l] = o(this, Si);
  return [(o(this, Gl) + (t[0] - e[0]) / 2 - n) / a, (o(this, Xl) + (t[1] - e[1]) / 2 - s) / l, (o(this, Gl) + (e[0] - t[0]) / 2 - n) / a, (o(this, Xl) + (e[1] - t[1]) / 2 - s) / l];
}, Y_ = function() {
  const [t, e, n, s] = o(this, Si), [a, l, u, h] = E(this, Ke, zd).call(this);
  return `M${(o(this, Ut)[2] - t) / n} ${(o(this, Ut)[3] - e) / s} L${(o(this, Ut)[4] - t) / n} ${(o(this, Ut)[5] - e) / s} L${a} ${l} L${u} ${h} L${(o(this, Ut)[16] - t) / n} ${(o(this, Ut)[17] - e) / s} L${(o(this, Ut)[14] - t) / n} ${(o(this, Ut)[15] - e) / s} Z`;
}, K_ = function(t) {
  const e = o(this, bs);
  t.push(`L${e[4]} ${e[5]} Z`);
}, Q_ = function(t) {
  const [e, n, s, a] = o(this, Si), l = o(this, Ut).subarray(4, 6), u = o(this, Ut).subarray(16, 18), [h, f, g, y] = E(this, Ke, zd).call(this);
  t.push(`L${(l[0] - e) / s} ${(l[1] - n) / a} L${h} ${f} L${g} ${y} L${(u[0] - e) / s} ${(u[1] - n) / a}`);
}, Z_ = function(t) {
  const e = o(this, Ut), [n, s, a, l] = o(this, Si), [u, h, f, g] = E(this, Ke, zd).call(this), y = new Float32Array(36);
  return y.set([NaN, NaN, NaN, NaN, (e[2] - n) / a, (e[3] - s) / l, NaN, NaN, NaN, NaN, (e[4] - n) / a, (e[5] - s) / l, NaN, NaN, NaN, NaN, u, h, NaN, NaN, NaN, NaN, f, g, NaN, NaN, NaN, NaN, (e[16] - n) / a, (e[17] - s) / l, NaN, NaN, NaN, NaN, (e[14] - n) / a, (e[15] - s) / l], 0), this.newFreeDrawOutline(y, t, o(this, Si), o(this, uh), o(this, lh), o(this, ch));
}, J_ = function(t, e) {
  const n = o(this, bs);
  return t.set([NaN, NaN, NaN, NaN, n[4], n[5]], e), e += 6;
}, tC = function(t, e) {
  const n = o(this, Ut).subarray(4, 6), s = o(this, Ut).subarray(16, 18), [a, l, u, h] = o(this, Si), [f, g, y, v] = E(this, Ke, zd).call(this);
  return t.set([NaN, NaN, NaN, NaN, (n[0] - a) / u, (n[1] - l) / h, NaN, NaN, NaN, NaN, f, g, NaN, NaN, NaN, NaN, y, v, NaN, NaN, NaN, NaN, (s[0] - a) / u, (s[1] - l) / h], e), e += 24;
}, b(_r, wp, 8), b(_r, gy, 2), b(_r, my, o(_r, wp) + o(_r, gy));
let Km = _r;
var dh, ql, Zr, Ap, bi, Sp, ze, yy, nC;
class eC extends dt {
  constructor(e, n, s, a, l, u) {
    super();
    b(this, yy);
    b(this, dh);
    b(this, ql, new Float32Array(4));
    b(this, Zr);
    b(this, Ap);
    b(this, bi);
    b(this, Sp);
    b(this, ze);
    w(this, ze, e), w(this, bi, n), w(this, dh, s), w(this, Sp, a), w(this, Zr, l), w(this, Ap, u), this.firstPoint = [NaN, NaN], this.lastPoint = [NaN, NaN], E(this, yy, nC).call(this, u);
    const [h, f, g, y] = o(this, ql);
    for (let v = 0, S = e.length; v < S; v += 2)
      e[v] = (e[v] - h) / g, e[v + 1] = (e[v + 1] - f) / y;
    for (let v = 0, S = n.length; v < S; v += 2)
      n[v] = (n[v] - h) / g, n[v + 1] = (n[v + 1] - f) / y;
  }
  toSVGPath() {
    const e = [`M${o(this, ze)[4]} ${o(this, ze)[5]}`];
    for (let n = 6, s = o(this, ze).length; n < s; n += 6) {
      if (isNaN(o(this, ze)[n])) {
        e.push(`L${o(this, ze)[n + 4]} ${o(this, ze)[n + 5]}`);
        continue;
      }
      e.push(`C${o(this, ze)[n]} ${o(this, ze)[n + 1]} ${o(this, ze)[n + 2]} ${o(this, ze)[n + 3]} ${o(this, ze)[n + 4]} ${o(this, ze)[n + 5]}`);
    }
    return e.push("Z"), e.join(" ");
  }
  serialize([e, n, s, a], l) {
    const u = s - e, h = a - n;
    let f, g;
    switch (l) {
      case 0:
        f = dt._rescale(o(this, ze), e, a, u, -h), g = dt._rescale(o(this, bi), e, a, u, -h);
        break;
      case 90:
        f = dt._rescaleAndSwap(o(this, ze), e, n, u, h), g = dt._rescaleAndSwap(o(this, bi), e, n, u, h);
        break;
      case 180:
        f = dt._rescale(o(this, ze), s, n, -u, h), g = dt._rescale(o(this, bi), s, n, -u, h);
        break;
      case 270:
        f = dt._rescaleAndSwap(o(this, ze), s, a, -u, -h), g = dt._rescaleAndSwap(o(this, bi), s, a, -u, -h);
        break;
    }
    return {
      outline: Array.from(f),
      points: [Array.from(g)]
    };
  }
  get box() {
    return o(this, ql);
  }
  newOutliner(e, n, s, a, l, u = 0) {
    return new Km(e, n, s, a, l, u);
  }
  getNewOutline(e, n) {
    const [s, a, l, u] = o(this, ql), [h, f, g, y] = o(this, dh), v = l * g, S = u * y, x = s * g + h, _ = a * y + f, k = this.newOutliner({
      x: o(this, bi)[0] * v + x,
      y: o(this, bi)[1] * S + _
    }, o(this, dh), o(this, Sp), e, o(this, Ap), n ?? o(this, Zr));
    for (let T = 2; T < o(this, bi).length; T += 2)
      k.add({
        x: o(this, bi)[T] * v + x,
        y: o(this, bi)[T + 1] * S + _
      });
    return k.getOutlines();
  }
}
dh = new WeakMap(), ql = new WeakMap(), Zr = new WeakMap(), Ap = new WeakMap(), bi = new WeakMap(), Sp = new WeakMap(), ze = new WeakMap(), yy = new WeakSet(), nC = function(e) {
  const n = o(this, ze);
  let s = n[4], a = n[5];
  const l = [s, a, s, a];
  let u = s, h = a, f = s, g = a;
  const y = e ? Math.max : Math.min, v = new Float32Array(4);
  for (let x = 6, _ = n.length; x < _; x += 6) {
    const k = n[x + 4], T = n[x + 5];
    isNaN(n[x]) ? (ft.pointBoundingBox(k, T, l), h > T ? (u = k, h = T) : h === T && (u = y(u, k)), g < T ? (f = k, g = T) : g === T && (f = y(f, k))) : (v[0] = v[1] = 1 / 0, v[2] = v[3] = -1 / 0, ft.bezierBoundingBox(s, a, ...n.slice(x, x + 6), v), ft.rectBoundingBox(v[0], v[1], v[2], v[3], l), h > v[1] ? (u = v[0], h = v[1]) : h === v[1] && (u = y(u, v[0])), g < v[3] ? (f = v[2], g = v[3]) : g === v[3] && (f = y(f, v[2]))), s = k, a = T;
  }
  const S = o(this, ql);
  S[0] = l[0] - o(this, Zr), S[1] = l[1] - o(this, Zr), S[2] = l[2] - l[0] + 2 * o(this, Zr), S[3] = l[3] - l[1] + 2 * o(this, Zr), this.firstPoint = [u, h], this.lastPoint = [f, g];
};
var bp, Ep, _p, pa, Es, $n, iC, Mm, sC, rC, Bw;
class Ow {
  constructor(t, e = 0, n = 0, s = !0) {
    b(this, $n);
    b(this, bp);
    b(this, Ep);
    b(this, _p);
    b(this, pa, []);
    b(this, Es, []);
    const a = [1 / 0, 1 / 0, -1 / 0, -1 / 0], l = 10 ** -4;
    for (const {
      x: _,
      y: k,
      width: T,
      height: R
    } of t) {
      const P = Math.floor((_ - e) / l) * l, L = Math.ceil((_ + T + e) / l) * l, D = Math.floor((k - e) / l) * l, N = Math.ceil((k + R + e) / l) * l, B = [P, D, N, !0], $ = [L, D, N, !1];
      o(this, pa).push(B, $), ft.rectBoundingBox(P, D, L, N, a);
    }
    const u = a[2] - a[0] + 2 * n, h = a[3] - a[1] + 2 * n, f = a[0] - n, g = a[1] - n;
    let y = s ? -1 / 0 : 1 / 0, v = 1 / 0;
    const S = o(this, pa).at(s ? -1 : -2), x = [S[0], S[2]];
    for (const _ of o(this, pa)) {
      const [k, T, R, P] = _;
      !P && s ? T < v ? (v = T, y = k) : T === v && (y = Math.max(y, k)) : P && !s && (T < v ? (v = T, y = k) : T === v && (y = Math.min(y, k))), _[0] = (k - f) / u, _[1] = (T - g) / h, _[2] = (R - g) / h;
    }
    w(this, bp, new Float32Array([f, g, u, h])), w(this, Ep, [y, v]), w(this, _p, x);
  }
  getOutlines() {
    o(this, pa).sort((e, n) => e[0] - n[0] || e[1] - n[1] || e[2] - n[2]);
    const t = [];
    for (const e of o(this, pa))
      e[3] ? (t.push(...E(this, $n, Bw).call(this, e)), E(this, $n, sC).call(this, e)) : (E(this, $n, rC).call(this, e), t.push(...E(this, $n, Bw).call(this, e)));
    return E(this, $n, iC).call(this, t);
  }
}
bp = new WeakMap(), Ep = new WeakMap(), _p = new WeakMap(), pa = new WeakMap(), Es = new WeakMap(), $n = new WeakSet(), iC = function(t) {
  const e = [], n = /* @__PURE__ */ new Set();
  for (const l of t) {
    const [u, h, f] = l;
    e.push([u, h, l], [u, f, l]);
  }
  e.sort((l, u) => l[1] - u[1] || l[0] - u[0]);
  for (let l = 0, u = e.length; l < u; l += 2) {
    const h = e[l][2], f = e[l + 1][2];
    h.push(f), f.push(h), n.add(h), n.add(f);
  }
  const s = [];
  let a;
  for (; n.size > 0; ) {
    const l = n.values().next().value;
    let [u, h, f, g, y] = l;
    n.delete(l);
    let v = u, S = h;
    for (a = [u, f], s.push(a); ; ) {
      let x;
      if (n.has(g))
        x = g;
      else if (n.has(y))
        x = y;
      else
        break;
      n.delete(x), [u, h, f, g, y] = x, v !== u && (a.push(v, S, u, S === h ? h : f), v = u), S = S === h ? f : h;
    }
    a.push(v, S);
  }
  return new VP(s, o(this, bp), o(this, Ep), o(this, _p));
}, Mm = function(t) {
  const e = o(this, Es);
  let n = 0, s = e.length - 1;
  for (; n <= s; ) {
    const a = n + s >> 1, l = e[a][0];
    if (l === t)
      return a;
    l < t ? n = a + 1 : s = a - 1;
  }
  return s + 1;
}, sC = function([, t, e]) {
  const n = E(this, $n, Mm).call(this, t);
  o(this, Es).splice(n, 0, [t, e]);
}, rC = function([, t, e]) {
  const n = E(this, $n, Mm).call(this, t);
  for (let s = n; s < o(this, Es).length; s++) {
    const [a, l] = o(this, Es)[s];
    if (a !== t)
      break;
    if (a === t && l === e) {
      o(this, Es).splice(s, 1);
      return;
    }
  }
  for (let s = n - 1; s >= 0; s--) {
    const [a, l] = o(this, Es)[s];
    if (a !== t)
      break;
    if (a === t && l === e) {
      o(this, Es).splice(s, 1);
      return;
    }
  }
}, Bw = function(t) {
  const [e, n, s] = t, a = [[e, n, s]], l = E(this, $n, Mm).call(this, s);
  for (let u = 0; u < l; u++) {
    const [h, f] = o(this, Es)[u];
    for (let g = 0, y = a.length; g < y; g++) {
      const [, v, S] = a[g];
      if (!(f <= v || S <= h)) {
        if (v >= h) {
          if (S > f)
            a[g][1] = f;
          else {
            if (y === 1)
              return [];
            a.splice(g, 1), g--, y--;
          }
          continue;
        }
        a[g][2] = h, S > f && a.push([e, f, S]);
      }
    }
  }
  return a;
};
var Cp, fh;
class VP extends dt {
  constructor(e, n, s, a) {
    super();
    b(this, Cp);
    b(this, fh);
    w(this, fh, e), w(this, Cp, n), this.firstPoint = s, this.lastPoint = a;
  }
  toSVGPath() {
    const e = [];
    for (const n of o(this, fh)) {
      let [s, a] = n;
      e.push(`M${s} ${a}`);
      for (let l = 2; l < n.length; l += 2) {
        const u = n[l], h = n[l + 1];
        u === s ? (e.push(`V${h}`), a = h) : h === a && (e.push(`H${u}`), s = u);
      }
      e.push("Z");
    }
    return e.join(" ");
  }
  serialize([e, n, s, a], l) {
    const u = [], h = s - e, f = a - n;
    for (const g of o(this, fh)) {
      const y = new Array(g.length);
      for (let v = 0; v < g.length; v += 2)
        y[v] = e + g[v] * h, y[v + 1] = a - g[v + 1] * f;
      u.push(y);
    }
    return u;
  }
  get box() {
    return o(this, Cp);
  }
  get classNamesForOutlining() {
    return ["highlightOutline"];
  }
}
Cp = new WeakMap(), fh = new WeakMap();
class zw extends Km {
  newFreeDrawOutline(t, e, n, s, a, l) {
    return new WP(t, e, n, s, a, l);
  }
}
class WP extends eC {
  newOutliner(t, e, n, s, a, l = 0) {
    return new zw(t, e, n, s, a, l);
  }
}
var ph, xp, Jr, Yl, Tp, ii, kp, Pp, Kl, Ei, _i, mn, gh, mh, Mn, yh, Zi, Rp, Ft, Uw, Lm, oC, aC, lC, Hw, Ya, es, Xc, cC, Dm, Im, uC, hC, dC, fC, pC;
const qt = class qt extends fe {
  constructor(e) {
    super({
      ...e,
      name: "highlightEditor"
    });
    b(this, Ft);
    b(this, ph, null);
    b(this, xp, 0);
    b(this, Jr);
    b(this, Yl, null);
    b(this, Tp, null);
    b(this, ii, null);
    b(this, kp, null);
    b(this, Pp, 0);
    b(this, Kl, null);
    b(this, Ei, null);
    b(this, _i, null);
    b(this, mn, !1);
    b(this, gh, null);
    b(this, mh, null);
    b(this, Mn, null);
    b(this, yh, "");
    b(this, Zi);
    b(this, Rp, "");
    this.color = e.color || qt._defaultColor, w(this, Zi, e.thickness || qt._defaultThickness), this.opacity = e.opacity || qt._defaultOpacity, w(this, Jr, e.boxes || null), w(this, Rp, e.methodOfCreation || ""), w(this, yh, e.text || ""), this._isDraggable = !1, this.defaultL10nId = "pdfjs-editor-highlight-editor", e.highlightId > -1 ? (w(this, mn, !0), E(this, Ft, Lm).call(this, e), E(this, Ft, Ya).call(this)) : o(this, Jr) && (w(this, ph, e.anchorNode), w(this, xp, e.anchorOffset), w(this, kp, e.focusNode), w(this, Pp, e.focusOffset), E(this, Ft, Uw).call(this), E(this, Ft, Ya).call(this), this.rotate(this.rotation)), this.annotationElementId || this._uiManager.a11yAlert("pdfjs-editor-highlight-added-alert");
  }
  static get _keyboardManager() {
    const e = qt.prototype;
    return It(this, "_keyboardManager", new Bp([[["ArrowLeft", "mac+ArrowLeft"], e._moveCaret, {
      args: [0]
    }], [["ArrowRight", "mac+ArrowRight"], e._moveCaret, {
      args: [1]
    }], [["ArrowUp", "mac+ArrowUp"], e._moveCaret, {
      args: [2]
    }], [["ArrowDown", "mac+ArrowDown"], e._moveCaret, {
      args: [3]
    }]]));
  }
  get telemetryInitialData() {
    return {
      action: "added",
      type: o(this, mn) ? "free_highlight" : "highlight",
      color: this._uiManager.getNonHCMColorName(this.color),
      thickness: o(this, Zi),
      methodOfCreation: o(this, Rp)
    };
  }
  get telemetryFinalData() {
    return {
      type: "highlight",
      color: this._uiManager.getNonHCMColorName(this.color)
    };
  }
  static computeTelemetryFinalData(e) {
    return {
      numberOfColors: e.get("color").size
    };
  }
  static initialize(e, n) {
    var s;
    fe.initialize(e, n), qt._defaultColor || (qt._defaultColor = ((s = n.highlightColors) == null ? void 0 : s.values().next().value) || "#fff066");
  }
  static updateDefaultParams(e, n) {
    switch (e) {
      case jt.HIGHLIGHT_COLOR:
        qt._defaultColor = n;
        break;
      case jt.HIGHLIGHT_THICKNESS:
        qt._defaultThickness = n;
        break;
    }
  }
  translateInPage(e, n) {
  }
  get toolbarPosition() {
    return o(this, mh);
  }
  get commentButtonPosition() {
    return o(this, gh);
  }
  updateParams(e, n) {
    switch (e) {
      case jt.HIGHLIGHT_COLOR:
        E(this, Ft, oC).call(this, n);
        break;
      case jt.HIGHLIGHT_THICKNESS:
        E(this, Ft, aC).call(this, n);
        break;
    }
  }
  static get defaultPropertiesToUpdate() {
    return [[jt.HIGHLIGHT_COLOR, qt._defaultColor], [jt.HIGHLIGHT_THICKNESS, qt._defaultThickness]];
  }
  get propertiesToUpdate() {
    return [[jt.HIGHLIGHT_COLOR, this.color || qt._defaultColor], [jt.HIGHLIGHT_THICKNESS, o(this, Zi) || qt._defaultThickness], [jt.HIGHLIGHT_FREE, o(this, mn)]];
  }
  onUpdatedColor() {
    var e, n;
    (e = this.parent) == null || e.drawLayer.updateProperties(o(this, _i), {
      root: {
        fill: this.color,
        "fill-opacity": this.opacity
      }
    }), (n = o(this, Tp)) == null || n.updateColor(this.color), super.onUpdatedColor();
  }
  get toolbarButtons() {
    return this._uiManager.highlightColors ? [["colorPicker", w(this, Tp, new Qd({
      editor: this
    }))]] : super.toolbarButtons;
  }
  disableEditing() {
    super.disableEditing(), this.div.classList.toggle("disabled", !0);
  }
  enableEditing() {
    super.enableEditing(), this.div.classList.toggle("disabled", !1);
  }
  fixAndSetPosition() {
    return super.fixAndSetPosition(E(this, Ft, Im).call(this));
  }
  getBaseTranslation() {
    return [0, 0];
  }
  getRect(e, n) {
    return super.getRect(e, n, E(this, Ft, Im).call(this));
  }
  onceAdded(e) {
    this.annotationElementId || this.parent.addUndoableEditor(this), e && this.div.focus();
  }
  remove() {
    E(this, Ft, Hw).call(this), this._reportTelemetry({
      action: "deleted"
    }), super.remove();
  }
  rebuild() {
    this.parent && (super.rebuild(), this.div !== null && (E(this, Ft, Ya).call(this), this.isAttachedToDOM || this.parent.add(this)));
  }
  setParent(e) {
    var s;
    let n = !1;
    this.parent && !e ? E(this, Ft, Hw).call(this) : e && (E(this, Ft, Ya).call(this, e), n = !this.parent && ((s = this.div) == null ? void 0 : s.classList.contains("selectedEditor"))), super.setParent(e), this.show(this._isVisible), n && this.select();
  }
  rotate(e) {
    var a, l, u;
    const {
      drawLayer: n
    } = this.parent;
    let s;
    o(this, mn) ? (e = (e - this.rotation + 360) % 360, s = E(a = qt, es, Xc).call(a, o(this, Ei).box, e)) : s = E(l = qt, es, Xc).call(l, [this.x, this.y, this.width, this.height], e), n.updateProperties(o(this, _i), {
      bbox: s,
      root: {
        "data-main-rotation": e
      }
    }), n.updateProperties(o(this, Mn), {
      bbox: E(u = qt, es, Xc).call(u, o(this, ii).box, e),
      root: {
        "data-main-rotation": e
      }
    });
  }
  render() {
    if (this.div)
      return this.div;
    const e = super.render();
    o(this, yh) && (e.setAttribute("aria-label", o(this, yh)), e.setAttribute("role", "mark")), o(this, mn) ? e.classList.add("free") : this.div.addEventListener("keydown", E(this, Ft, cC).bind(this), {
      signal: this._uiManager._signal
    });
    const n = w(this, Kl, document.createElement("div"));
    return e.append(n), n.setAttribute("aria-hidden", "true"), n.className = "internal", n.style.clipPath = o(this, Yl), this.setDims(this.width, this.height), I1(this, o(this, Kl), ["pointerover", "pointerleave"]), this.enableEditing(), e;
  }
  pointerover() {
    var e;
    this.isSelected || (e = this.parent) == null || e.drawLayer.updateProperties(o(this, Mn), {
      rootClass: {
        hovered: !0
      }
    });
  }
  pointerleave() {
    var e;
    this.isSelected || (e = this.parent) == null || e.drawLayer.updateProperties(o(this, Mn), {
      rootClass: {
        hovered: !1
      }
    });
  }
  _moveCaret(e) {
    switch (this.parent.unselect(this), e) {
      case 0:
      case 2:
        E(this, Ft, Dm).call(this, !0);
        break;
      case 1:
      case 3:
        E(this, Ft, Dm).call(this, !1);
        break;
    }
  }
  select() {
    var e;
    super.select(), o(this, Mn) && ((e = this.parent) == null || e.drawLayer.updateProperties(o(this, Mn), {
      rootClass: {
        hovered: !1,
        selected: !0
      }
    }));
  }
  unselect() {
    var e;
    super.unselect(), o(this, Mn) && ((e = this.parent) == null || e.drawLayer.updateProperties(o(this, Mn), {
      rootClass: {
        selected: !1
      }
    }), o(this, mn) || E(this, Ft, Dm).call(this, !1));
  }
  get _mustFixPosition() {
    return !o(this, mn);
  }
  show(e = this._isVisible) {
    super.show(e), this.parent && (this.parent.drawLayer.updateProperties(o(this, _i), {
      rootClass: {
        hidden: !e
      }
    }), this.parent.drawLayer.updateProperties(o(this, Mn), {
      rootClass: {
        hidden: !e
      }
    }));
  }
  static startHighlighting(e, n, {
    target: s,
    x: a,
    y: l
  }) {
    const {
      x: u,
      y: h,
      width: f,
      height: g
    } = s.getBoundingClientRect(), y = new AbortController(), v = e.combinedSignal(y), S = (x) => {
      y.abort(), E(this, es, fC).call(this, e, x);
    };
    window.addEventListener("blur", S, {
      signal: v
    }), window.addEventListener("pointerup", S, {
      signal: v
    }), window.addEventListener("pointerdown", Ce, {
      capture: !0,
      passive: !1,
      signal: v
    }), window.addEventListener("contextmenu", Li, {
      signal: v
    }), s.addEventListener("pointermove", E(this, es, dC).bind(this, e), {
      signal: v
    }), this._freeHighlight = new zw({
      x: a,
      y: l
    }, [u, h, f, g], e.scale, this._defaultThickness / 2, n, 1e-3), {
      id: this._freeHighlightId,
      clipPathId: this._freeHighlightClipId
    } = e.drawLayer.draw({
      bbox: [0, 0, 1, 1],
      root: {
        viewBox: "0 0 1 1",
        fill: this._defaultColor,
        "fill-opacity": this._defaultOpacity
      },
      rootClass: {
        highlight: !0,
        free: !0
      },
      path: {
        d: this._freeHighlight.toSVGPath()
      }
    }, !0, !0);
  }
  static async deserialize(e, n, s) {
    var _, k, T, R;
    let a = null;
    if (e instanceof j_) {
      const {
        data: {
          quadPoints: P,
          rect: L,
          rotation: D,
          id: N,
          color: B,
          opacity: $,
          popupRef: V,
          richText: X,
          contentsObj: q,
          creationDate: it,
          modificationDate: lt
        },
        parent: {
          page: {
            pageNumber: rt
          }
        }
      } = e;
      a = e = {
        annotationType: Pt.HIGHLIGHT,
        color: Array.from(B),
        opacity: $,
        quadPoints: P,
        boxes: null,
        pageIndex: rt - 1,
        rect: L.slice(0),
        rotation: D,
        annotationElementId: N,
        id: N,
        deleted: !1,
        popupRef: V,
        richText: X,
        comment: (q == null ? void 0 : q.str) || null,
        creationDate: it,
        modificationDate: lt
      };
    } else if (e instanceof AA) {
      const {
        data: {
          inkLists: P,
          rect: L,
          rotation: D,
          id: N,
          color: B,
          borderStyle: {
            rawWidth: $
          },
          popupRef: V,
          richText: X,
          contentsObj: q,
          creationDate: it,
          modificationDate: lt
        },
        parent: {
          page: {
            pageNumber: rt
          }
        }
      } = e;
      a = e = {
        annotationType: Pt.HIGHLIGHT,
        color: Array.from(B),
        thickness: $,
        inkLists: P,
        boxes: null,
        pageIndex: rt - 1,
        rect: L.slice(0),
        rotation: D,
        annotationElementId: N,
        id: N,
        deleted: !1,
        popupRef: V,
        richText: X,
        comment: (q == null ? void 0 : q.str) || null,
        creationDate: it,
        modificationDate: lt
      };
    }
    const {
      color: l,
      quadPoints: u,
      inkLists: h,
      opacity: f
    } = e, g = await super.deserialize(e, n, s);
    g.color = ft.makeHexColor(...l), g.opacity = f || 1, h && w(g, Zi, e.thickness), g._initialData = a, e.comment && g.setCommentData(e);
    const [y, v] = g.pageDimensions, [S, x] = g.pageTranslation;
    if (u) {
      const P = w(g, Jr, []);
      for (let L = 0; L < u.length; L += 8)
        P.push({
          x: (u[L] - S) / y,
          y: 1 - (u[L + 1] - x) / v,
          width: (u[L + 2] - u[L]) / y,
          height: (u[L + 1] - u[L + 5]) / v
        });
      E(_ = g, Ft, Uw).call(_), E(k = g, Ft, Ya).call(k), g.rotate(g.rotation);
    } else if (h) {
      w(g, mn, !0);
      const P = h[0], L = {
        x: P[0] - S,
        y: v - (P[1] - x)
      }, D = new zw(L, [0, 0, y, v], 1, o(g, Zi) / 2, !0, 1e-3);
      for (let $ = 0, V = P.length; $ < V; $ += 2)
        L.x = P[$] - S, L.y = v - (P[$ + 1] - x), D.add(L);
      const {
        id: N,
        clipPathId: B
      } = n.drawLayer.draw({
        bbox: [0, 0, 1, 1],
        root: {
          viewBox: "0 0 1 1",
          fill: g.color,
          "fill-opacity": g._defaultOpacity
        },
        rootClass: {
          highlight: !0,
          free: !0
        },
        path: {
          d: D.toSVGPath()
        }
      }, !0, !0);
      E(T = g, Ft, Lm).call(T, {
        highlightOutlines: D.getOutlines(),
        highlightId: N,
        clipPathId: B
      }), E(R = g, Ft, Ya).call(R), g.rotate(g.parentRotation);
    }
    return g;
  }
  serialize(e = !1) {
    if (this.isEmpty() || e)
      return null;
    if (this.deleted)
      return this.serializeDeleted();
    const n = fe._colorManager.convert(this._uiManager.getNonHCMColor(this.color)), s = super.serialize(e);
    return Object.assign(s, {
      color: n,
      opacity: this.opacity,
      thickness: o(this, Zi),
      quadPoints: E(this, Ft, uC).call(this),
      outlines: E(this, Ft, hC).call(this, s.rect)
    }), this.addComment(s), this.annotationElementId && !E(this, Ft, pC).call(this, s) ? null : (s.id = this.annotationElementId, s);
  }
  renderAnnotationElement(e) {
    return this.deleted ? (e.hide(), null) : (e.updateEdited({
      rect: this.getPDFRect(),
      popup: this.comment
    }), null);
  }
  static canCreateNewEmptyEditor() {
    return !1;
  }
};
ph = new WeakMap(), xp = new WeakMap(), Jr = new WeakMap(), Yl = new WeakMap(), Tp = new WeakMap(), ii = new WeakMap(), kp = new WeakMap(), Pp = new WeakMap(), Kl = new WeakMap(), Ei = new WeakMap(), _i = new WeakMap(), mn = new WeakMap(), gh = new WeakMap(), mh = new WeakMap(), Mn = new WeakMap(), yh = new WeakMap(), Zi = new WeakMap(), Rp = new WeakMap(), Ft = new WeakSet(), Uw = function() {
  const e = new Ow(o(this, Jr), 1e-3);
  w(this, Ei, e.getOutlines()), [this.x, this.y, this.width, this.height] = o(this, Ei).box;
  const n = new Ow(o(this, Jr), 25e-4, 1e-3, this._uiManager.direction === "ltr");
  w(this, ii, n.getOutlines());
  const {
    firstPoint: s
  } = o(this, Ei);
  w(this, gh, [(s[0] - this.x) / this.width, (s[1] - this.y) / this.height]);
  const {
    lastPoint: a
  } = o(this, ii);
  w(this, mh, [(a[0] - this.x) / this.width, (a[1] - this.y) / this.height]);
}, Lm = function({
  highlightOutlines: e,
  highlightId: n,
  clipPathId: s
}) {
  var v, S;
  if (w(this, Ei, e), w(this, ii, e.getNewOutline(o(this, Zi) / 2 + 1.5, 25e-4)), n >= 0)
    w(this, _i, n), w(this, Yl, s), this.parent.drawLayer.finalizeDraw(n, {
      bbox: e.box,
      path: {
        d: e.toSVGPath()
      }
    }), w(this, Mn, this.parent.drawLayer.drawOutline({
      rootClass: {
        highlightOutline: !0,
        free: !0
      },
      bbox: o(this, ii).box,
      path: {
        d: o(this, ii).toSVGPath()
      }
    }, !0));
  else if (this.parent) {
    const x = this.parent.viewport.rotation;
    this.parent.drawLayer.updateProperties(o(this, _i), {
      bbox: E(v = qt, es, Xc).call(v, o(this, Ei).box, (x - this.rotation + 360) % 360),
      path: {
        d: e.toSVGPath()
      }
    }), this.parent.drawLayer.updateProperties(o(this, Mn), {
      bbox: E(S = qt, es, Xc).call(S, o(this, ii).box, x),
      path: {
        d: o(this, ii).toSVGPath()
      }
    });
  }
  const [l, u, h, f] = e.box;
  switch (this.rotation) {
    case 0:
      this.x = l, this.y = u, this.width = h, this.height = f;
      break;
    case 90: {
      const [x, _] = this.parentDimensions;
      this.x = u, this.y = 1 - l, this.width = h * _ / x, this.height = f * x / _;
      break;
    }
    case 180:
      this.x = 1 - l, this.y = 1 - u, this.width = h, this.height = f;
      break;
    case 270: {
      const [x, _] = this.parentDimensions;
      this.x = 1 - u, this.y = l, this.width = h * _ / x, this.height = f * x / _;
      break;
    }
  }
  const {
    firstPoint: g
  } = e;
  w(this, gh, [(g[0] - l) / h, (g[1] - u) / f]);
  const {
    lastPoint: y
  } = o(this, ii);
  w(this, mh, [(y[0] - l) / h, (y[1] - u) / f]);
}, oC = function(e) {
  const n = (l, u) => {
    this.color = l, this.opacity = u, this.onUpdatedColor();
  }, s = this.color, a = this.opacity;
  this.addCommands({
    cmd: n.bind(this, e, qt._defaultOpacity),
    undo: n.bind(this, s, a),
    post: this._uiManager.updateUI.bind(this._uiManager, this),
    mustExec: !0,
    type: jt.HIGHLIGHT_COLOR,
    overwriteIfSameType: !0,
    keepUndo: !0
  }), this._reportTelemetry({
    action: "color_changed",
    color: this._uiManager.getNonHCMColorName(e)
  }, !0);
}, aC = function(e) {
  const n = o(this, Zi), s = (a) => {
    w(this, Zi, a), E(this, Ft, lC).call(this, a);
  };
  this.addCommands({
    cmd: s.bind(this, e),
    undo: s.bind(this, n),
    post: this._uiManager.updateUI.bind(this._uiManager, this),
    mustExec: !0,
    type: jt.INK_THICKNESS,
    overwriteIfSameType: !0,
    keepUndo: !0
  }), this._reportTelemetry({
    action: "thickness_changed",
    thickness: e
  }, !0);
}, lC = function(e) {
  o(this, mn) && (E(this, Ft, Lm).call(this, {
    highlightOutlines: o(this, Ei).getNewOutline(e / 2)
  }), this.fixAndSetPosition(), this.setDims(this.width, this.height));
}, Hw = function() {
  o(this, _i) === null || !this.parent || (this.parent.drawLayer.remove(o(this, _i)), w(this, _i, null), this.parent.drawLayer.remove(o(this, Mn)), w(this, Mn, null));
}, Ya = function(e = this.parent) {
  o(this, _i) === null && ({
    id: Ge(this, _i)._,
    clipPathId: Ge(this, Yl)._
  } = e.drawLayer.draw({
    bbox: o(this, Ei).box,
    root: {
      viewBox: "0 0 1 1",
      fill: this.color,
      "fill-opacity": this.opacity
    },
    rootClass: {
      highlight: !0,
      free: o(this, mn)
    },
    path: {
      d: o(this, Ei).toSVGPath()
    }
  }, !1, !0), w(this, Mn, e.drawLayer.drawOutline({
    rootClass: {
      highlightOutline: !0,
      free: o(this, mn)
    },
    bbox: o(this, ii).box,
    path: {
      d: o(this, ii).toSVGPath()
    }
  }, o(this, mn))), o(this, Kl) && (o(this, Kl).style.clipPath = o(this, Yl)));
}, es = new WeakSet(), Xc = function([e, n, s, a], l) {
  switch (l) {
    case 90:
      return [1 - n - a, e, a, s];
    case 180:
      return [1 - e - s, 1 - n - a, s, a];
    case 270:
      return [n, 1 - e - s, a, s];
  }
  return [e, n, s, a];
}, cC = function(e) {
  qt._keyboardManager.exec(this, e);
}, Dm = function(e) {
  if (!o(this, ph))
    return;
  const n = window.getSelection();
  e ? n.setPosition(o(this, ph), o(this, xp)) : n.setPosition(o(this, kp), o(this, Pp));
}, Im = function() {
  return o(this, mn) ? this.rotation : 0;
}, uC = function() {
  if (o(this, mn))
    return null;
  const [e, n] = this.pageDimensions, [s, a] = this.pageTranslation, l = o(this, Jr), u = new Float32Array(l.length * 8);
  let h = 0;
  for (const {
    x: f,
    y: g,
    width: y,
    height: v
  } of l) {
    const S = f * e + s, x = (1 - g) * n + a;
    u[h] = u[h + 4] = S, u[h + 1] = u[h + 3] = x, u[h + 2] = u[h + 6] = S + y * e, u[h + 5] = u[h + 7] = x - v * n, h += 8;
  }
  return u;
}, hC = function(e) {
  return o(this, Ei).serialize(e, E(this, Ft, Im).call(this));
}, dC = function(e, n) {
  this._freeHighlight.add(n) && e.drawLayer.updateProperties(this._freeHighlightId, {
    path: {
      d: this._freeHighlight.toSVGPath()
    }
  });
}, fC = function(e, n) {
  this._freeHighlight.isEmpty() ? e.drawLayer.remove(this._freeHighlightId) : e.createAndAddNewEditor(n, !1, {
    highlightId: this._freeHighlightId,
    highlightOutlines: this._freeHighlight.getOutlines(),
    clipPathId: this._freeHighlightClipId,
    methodOfCreation: "main_toolbar"
  }), this._freeHighlightId = -1, this._freeHighlight = null, this._freeHighlightClipId = "";
}, pC = function(e) {
  const {
    color: n
  } = this._initialData;
  return this.hasEditedComment || e.color.some((s, a) => s !== n[a]);
}, b(qt, es), Z(qt, "_defaultColor", null), Z(qt, "_defaultOpacity", 1), Z(qt, "_defaultThickness", 12), Z(qt, "_type", "highlight"), Z(qt, "_editorType", Pt.HIGHLIGHT), Z(qt, "_freeHighlightId", -1), Z(qt, "_freeHighlight", null), Z(qt, "_freeHighlightClipId", "");
let Qm = qt;
var Ql;
class gC {
  constructor() {
    b(this, Ql, /* @__PURE__ */ Object.create(null));
  }
  updateProperty(t, e) {
    this[t] = e, this.updateSVGProperty(t, e);
  }
  updateProperties(t) {
    if (t)
      for (const [e, n] of Object.entries(t))
        e.startsWith("_") || this.updateProperty(e, n);
  }
  updateSVGProperty(t, e) {
    o(this, Ql)[t] = e;
  }
  toSVGProperties() {
    const t = o(this, Ql);
    return w(this, Ql, /* @__PURE__ */ Object.create(null)), {
      root: t
    };
  }
  reset() {
    w(this, Ql, /* @__PURE__ */ Object.create(null));
  }
  updateAll(t = this) {
    this.updateProperties(t);
  }
  clone() {
    ce("Not implemented");
  }
}
Ql = new WeakMap();
var Ci, vh, sn, Zl, Jl, ga, ma, ya, tc, Wt, jw, $w, Vw, Ud, mC, Fm, Hd, qc;
const gt = class gt extends fe {
  constructor(e) {
    super(e);
    b(this, Wt);
    b(this, Ci, null);
    b(this, vh);
    Z(this, "_colorPicker", null);
    Z(this, "_drawId", null);
    w(this, vh, e.mustBeCommitted || !1), this._addOutlines(e);
  }
  onUpdatedColor() {
    var e;
    (e = this._colorPicker) == null || e.update(this.color), super.onUpdatedColor();
  }
  _addOutlines(e) {
    e.drawOutlines && (E(this, Wt, jw).call(this, e), E(this, Wt, Ud).call(this));
  }
  static _mergeSVGProperties(e, n) {
    const s = new Set(Object.keys(e));
    for (const [a, l] of Object.entries(n))
      s.has(a) ? Object.assign(e[a], l) : e[a] = l;
    return e;
  }
  static getDefaultDrawingOptions(e) {
    ce("Not implemented");
  }
  static get typesMap() {
    ce("Not implemented");
  }
  static get isDrawer() {
    return !0;
  }
  static get supportMultipleDrawings() {
    return !1;
  }
  static updateDefaultParams(e, n) {
    const s = this.typesMap.get(e);
    s && this._defaultDrawingOptions.updateProperty(s, n), this._currentParent && (o(gt, sn).updateProperty(s, n), this._currentParent.drawLayer.updateProperties(this._currentDrawId, this._defaultDrawingOptions.toSVGProperties()));
  }
  updateParams(e, n) {
    const s = this.constructor.typesMap.get(e);
    s && this._updateProperty(e, s, n);
  }
  static get defaultPropertiesToUpdate() {
    const e = [], n = this._defaultDrawingOptions;
    for (const [s, a] of this.typesMap)
      e.push([s, n[a]]);
    return e;
  }
  get propertiesToUpdate() {
    const e = [], {
      _drawingOptions: n
    } = this;
    for (const [s, a] of this.constructor.typesMap)
      e.push([s, n[a]]);
    return e;
  }
  _updateProperty(e, n, s) {
    const a = this._drawingOptions, l = a[n], u = (h) => {
      var g;
      a.updateProperty(n, h);
      const f = o(this, Ci).updateProperty(n, h);
      f && E(this, Wt, Hd).call(this, f), (g = this.parent) == null || g.drawLayer.updateProperties(this._drawId, a.toSVGProperties()), e === this.colorType && this.onUpdatedColor();
    };
    this.addCommands({
      cmd: u.bind(this, s),
      undo: u.bind(this, l),
      post: this._uiManager.updateUI.bind(this._uiManager, this),
      mustExec: !0,
      type: e,
      overwriteIfSameType: !0,
      keepUndo: !0
    });
  }
  _onResizing() {
    var e;
    (e = this.parent) == null || e.drawLayer.updateProperties(this._drawId, gt._mergeSVGProperties(o(this, Ci).getPathResizingSVGProperties(E(this, Wt, Fm).call(this)), {
      bbox: E(this, Wt, qc).call(this)
    }));
  }
  _onResized() {
    var e;
    (e = this.parent) == null || e.drawLayer.updateProperties(this._drawId, gt._mergeSVGProperties(o(this, Ci).getPathResizedSVGProperties(E(this, Wt, Fm).call(this)), {
      bbox: E(this, Wt, qc).call(this)
    }));
  }
  _onTranslating(e, n) {
    var s;
    (s = this.parent) == null || s.drawLayer.updateProperties(this._drawId, {
      bbox: E(this, Wt, qc).call(this)
    });
  }
  _onTranslated() {
    var e;
    (e = this.parent) == null || e.drawLayer.updateProperties(this._drawId, gt._mergeSVGProperties(o(this, Ci).getPathTranslatedSVGProperties(E(this, Wt, Fm).call(this), this.parentDimensions), {
      bbox: E(this, Wt, qc).call(this)
    }));
  }
  _onStartDragging() {
    var e;
    (e = this.parent) == null || e.drawLayer.updateProperties(this._drawId, {
      rootClass: {
        moving: !0
      }
    });
  }
  _onStopDragging() {
    var e;
    (e = this.parent) == null || e.drawLayer.updateProperties(this._drawId, {
      rootClass: {
        moving: !1
      }
    });
  }
  commit() {
    super.commit(), this.disableEditMode(), this.disableEditing();
  }
  disableEditing() {
    super.disableEditing(), this.div.classList.toggle("disabled", !0);
  }
  enableEditing() {
    super.enableEditing(), this.div.classList.toggle("disabled", !1);
  }
  getBaseTranslation() {
    return [0, 0];
  }
  get isResizable() {
    return !0;
  }
  onceAdded(e) {
    this.annotationElementId || this.parent.addUndoableEditor(this), this._isDraggable = !0, o(this, vh) && (w(this, vh, !1), this.commit(), this.parent.setSelected(this), e && this.isOnScreen && this.div.focus());
  }
  remove() {
    E(this, Wt, Vw).call(this), super.remove();
  }
  rebuild() {
    this.parent && (super.rebuild(), this.div !== null && (E(this, Wt, Ud).call(this), E(this, Wt, Hd).call(this, o(this, Ci).box), this.isAttachedToDOM || this.parent.add(this)));
  }
  setParent(e) {
    var s;
    let n = !1;
    this.parent && !e ? (this._uiManager.removeShouldRescale(this), E(this, Wt, Vw).call(this)) : e && (this._uiManager.addShouldRescale(this), E(this, Wt, Ud).call(this, e), n = !this.parent && ((s = this.div) == null ? void 0 : s.classList.contains("selectedEditor"))), super.setParent(e), n && this.select();
  }
  rotate() {
    this.parent && this.parent.drawLayer.updateProperties(this._drawId, gt._mergeSVGProperties({
      bbox: E(this, Wt, qc).call(this)
    }, o(this, Ci).updateRotation((this.parentRotation - this.rotation + 360) % 360)));
  }
  onScaleChanging() {
    this.parent && E(this, Wt, Hd).call(this, o(this, Ci).updateParentDimensions(this.parentDimensions, this.parent.scale));
  }
  static onScaleChangingWhenDrawing() {
  }
  render() {
    if (this.div)
      return this.div;
    let e, n;
    this._isCopy && (e = this.x, n = this.y);
    const s = super.render();
    s.classList.add("draw");
    const a = document.createElement("div");
    return s.append(a), a.setAttribute("aria-hidden", "true"), a.className = "internal", this.setDims(), this._uiManager.addShouldRescale(this), this.disableEditing(), this._isCopy && this._moveAfterPaste(e, n), s;
  }
  static createDrawerInstance(e, n, s, a, l) {
    ce("Not implemented");
  }
  static startDrawing(e, n, s, a) {
    var k;
    const {
      target: l,
      offsetX: u,
      offsetY: h,
      pointerId: f,
      pointerType: g
    } = a;
    if (o(gt, ma) && o(gt, ma) !== g)
      return;
    const {
      viewport: {
        rotation: y
      }
    } = e, {
      width: v,
      height: S
    } = l.getBoundingClientRect(), x = w(gt, Zl, new AbortController()), _ = e.combinedSignal(x);
    if (o(gt, ga) || w(gt, ga, f), o(gt, ma) ?? w(gt, ma, g), window.addEventListener("pointerup", (T) => {
      var R;
      o(gt, ga) === T.pointerId ? this._endDraw(T) : (R = o(gt, ya)) == null || R.delete(T.pointerId);
    }, {
      signal: _
    }), window.addEventListener("pointercancel", (T) => {
      var R;
      o(gt, ga) === T.pointerId ? this._currentParent.endDrawingSession() : (R = o(gt, ya)) == null || R.delete(T.pointerId);
    }, {
      signal: _
    }), window.addEventListener("pointerdown", (T) => {
      o(gt, ma) === T.pointerType && ((o(gt, ya) || w(gt, ya, /* @__PURE__ */ new Set())).add(T.pointerId), o(gt, sn).isCancellable() && (o(gt, sn).removeLastElement(), o(gt, sn).isEmpty() ? this._currentParent.endDrawingSession(!0) : this._endDraw(null)));
    }, {
      capture: !0,
      passive: !1,
      signal: _
    }), window.addEventListener("contextmenu", Li, {
      signal: _
    }), l.addEventListener("pointermove", this._drawMove.bind(this), {
      signal: _
    }), l.addEventListener("touchmove", (T) => {
      T.timeStamp === o(gt, tc) && Ce(T);
    }, {
      signal: _
    }), e.toggleDrawing(), (k = n._editorUndoBar) == null || k.hide(), o(gt, sn)) {
      e.drawLayer.updateProperties(this._currentDrawId, o(gt, sn).startNew(u, h, v, S, y));
      return;
    }
    n.updateUIForDefaultProperties(this), w(gt, sn, this.createDrawerInstance(u, h, v, S, y)), w(gt, Jl, this.getDefaultDrawingOptions()), this._currentParent = e, {
      id: this._currentDrawId
    } = e.drawLayer.draw(this._mergeSVGProperties(o(gt, Jl).toSVGProperties(), o(gt, sn).defaultSVGProperties), !0, !1);
  }
  static _drawMove(e) {
    var l;
    if (w(gt, tc, -1), !o(gt, sn))
      return;
    const {
      offsetX: n,
      offsetY: s,
      pointerId: a
    } = e;
    if (o(gt, ga) === a) {
      if (((l = o(gt, ya)) == null ? void 0 : l.size) >= 1) {
        this._endDraw(e);
        return;
      }
      this._currentParent.drawLayer.updateProperties(this._currentDrawId, o(gt, sn).add(n, s)), w(gt, tc, e.timeStamp), Ce(e);
    }
  }
  static _cleanup(e) {
    e && (this._currentDrawId = -1, this._currentParent = null, w(gt, sn, null), w(gt, Jl, null), w(gt, ma, null), w(gt, tc, NaN)), o(gt, Zl) && (o(gt, Zl).abort(), w(gt, Zl, null), w(gt, ga, NaN), w(gt, ya, null));
  }
  static _endDraw(e) {
    const n = this._currentParent;
    if (n) {
      if (n.toggleDrawing(!0), this._cleanup(!1), (e == null ? void 0 : e.target) === n.div && n.drawLayer.updateProperties(this._currentDrawId, o(gt, sn).end(e.offsetX, e.offsetY)), this.supportMultipleDrawings) {
        const s = o(gt, sn), a = this._currentDrawId, l = s.getLastElement();
        n.addCommands({
          cmd: () => {
            n.drawLayer.updateProperties(a, s.setLastElement(l));
          },
          undo: () => {
            n.drawLayer.updateProperties(a, s.removeLastElement());
          },
          mustExec: !1,
          type: jt.DRAW_STEP
        });
        return;
      }
      this.endDrawing(!1);
    }
  }
  static endDrawing(e) {
    const n = this._currentParent;
    if (!n)
      return null;
    if (n.toggleDrawing(!0), n.cleanUndoStack(jt.DRAW_STEP), !o(gt, sn).isEmpty()) {
      const {
        pageDimensions: [s, a],
        scale: l
      } = n, u = n.createAndAddNewEditor({
        offsetX: 0,
        offsetY: 0
      }, !1, {
        drawId: this._currentDrawId,
        drawOutlines: o(gt, sn).getOutlines(s * l, a * l, l, this._INNER_MARGIN),
        drawingOptions: o(gt, Jl),
        mustBeCommitted: !e
      });
      return this._cleanup(!0), u;
    }
    return n.drawLayer.remove(this._currentDrawId), this._cleanup(!0), null;
  }
  createDrawingOptions(e) {
  }
  static deserializeDraw(e, n, s, a, l, u) {
    ce("Not implemented");
  }
  static async deserialize(e, n, s) {
    var y, v;
    const {
      rawDims: {
        pageWidth: a,
        pageHeight: l,
        pageX: u,
        pageY: h
      }
    } = n.viewport, f = this.deserializeDraw(u, h, a, l, this._INNER_MARGIN, e), g = await super.deserialize(e, n, s);
    return g.createDrawingOptions(e), E(y = g, Wt, jw).call(y, {
      drawOutlines: f
    }), E(v = g, Wt, Ud).call(v), g.onScaleChanging(), g.rotate(), g;
  }
  serializeDraw(e) {
    const [n, s] = this.pageTranslation, [a, l] = this.pageDimensions;
    return o(this, Ci).serialize([n, s, a, l], e);
  }
  renderAnnotationElement(e) {
    return e.updateEdited({
      rect: this.getPDFRect()
    }), null;
  }
  static canCreateNewEmptyEditor() {
    return !1;
  }
};
Ci = new WeakMap(), vh = new WeakMap(), sn = new WeakMap(), Zl = new WeakMap(), Jl = new WeakMap(), ga = new WeakMap(), ma = new WeakMap(), ya = new WeakMap(), tc = new WeakMap(), Wt = new WeakSet(), jw = function({
  drawOutlines: e,
  drawId: n,
  drawingOptions: s
}) {
  w(this, Ci, e), this._drawingOptions || (this._drawingOptions = s), this.annotationElementId || this._uiManager.a11yAlert(`pdfjs-editor-${this.editorType}-added-alert`), n >= 0 ? (this._drawId = n, this.parent.drawLayer.finalizeDraw(n, e.defaultProperties)) : this._drawId = E(this, Wt, $w).call(this, e, this.parent), E(this, Wt, Hd).call(this, e.box);
}, $w = function(e, n) {
  const {
    id: s
  } = n.drawLayer.draw(gt._mergeSVGProperties(this._drawingOptions.toSVGProperties(), e.defaultSVGProperties), !1, !1);
  return s;
}, Vw = function() {
  this._drawId === null || !this.parent || (this.parent.drawLayer.remove(this._drawId), this._drawId = null, this._drawingOptions.reset());
}, Ud = function(e = this.parent) {
  if (!(this._drawId !== null && this.parent === e)) {
    if (this._drawId !== null) {
      this.parent.drawLayer.updateParent(this._drawId, e.drawLayer);
      return;
    }
    this._drawingOptions.updateAll(), this._drawId = E(this, Wt, $w).call(this, o(this, Ci), e);
  }
}, mC = function([e, n, s, a]) {
  const {
    parentDimensions: [l, u],
    rotation: h
  } = this;
  switch (h) {
    case 90:
      return [n, 1 - e, s * (u / l), a * (l / u)];
    case 180:
      return [1 - e, 1 - n, s, a];
    case 270:
      return [1 - n, e, s * (u / l), a * (l / u)];
    default:
      return [e, n, s, a];
  }
}, Fm = function() {
  const {
    x: e,
    y: n,
    width: s,
    height: a,
    parentDimensions: [l, u],
    rotation: h
  } = this;
  switch (h) {
    case 90:
      return [1 - n, e, s * (l / u), a * (u / l)];
    case 180:
      return [1 - e, 1 - n, s, a];
    case 270:
      return [n, 1 - e, s * (l / u), a * (u / l)];
    default:
      return [e, n, s, a];
  }
}, Hd = function(e) {
  [this.x, this.y, this.width, this.height] = E(this, Wt, mC).call(this, e), this.div && (this.fixAndSetPosition(), this.setDims()), this._onResized();
}, qc = function() {
  const {
    x: e,
    y: n,
    width: s,
    height: a,
    rotation: l,
    parentRotation: u,
    parentDimensions: [h, f]
  } = this;
  switch ((l * 4 + u) / 90) {
    case 1:
      return [1 - n - a, e, a, s];
    case 2:
      return [1 - e - s, 1 - n - a, s, a];
    case 3:
      return [n, 1 - e - s, a, s];
    case 4:
      return [e, n - s * (h / f), a * (f / h), s * (h / f)];
    case 5:
      return [1 - n, e, s * (h / f), a * (f / h)];
    case 6:
      return [1 - e - a * (f / h), 1 - n, a * (f / h), s * (h / f)];
    case 7:
      return [n - s * (h / f), 1 - e - a * (f / h), s * (h / f), a * (f / h)];
    case 8:
      return [e - s, n - a, s, a];
    case 9:
      return [1 - n, e - s, a, s];
    case 10:
      return [1 - e, 1 - n, s, a];
    case 11:
      return [n - a, 1 - e, a, s];
    case 12:
      return [e - a * (f / h), n, a * (f / h), s * (h / f)];
    case 13:
      return [1 - n - s * (h / f), e - a * (f / h), s * (h / f), a * (f / h)];
    case 14:
      return [1 - e, 1 - n - s * (h / f), a * (f / h), s * (h / f)];
    case 15:
      return [n, 1 - e, s * (h / f), a * (f / h)];
    default:
      return [e, n, s, a];
  }
}, Z(gt, "_currentDrawId", -1), Z(gt, "_currentParent", null), b(gt, sn, null), b(gt, Zl, null), b(gt, Jl, null), b(gt, ga, NaN), b(gt, ma, null), b(gt, ya, null), b(gt, tc, NaN), Z(gt, "_INNER_MARGIN", 3);
let Zm = gt;
var ir, rn, on, ec, wh, Bn, yn, Ji, nc, ic, sc, Ah, Nm;
class GP {
  constructor(t, e, n, s, a, l) {
    b(this, Ah);
    b(this, ir, new Float64Array(6));
    b(this, rn);
    b(this, on);
    b(this, ec);
    b(this, wh);
    b(this, Bn);
    b(this, yn, "");
    b(this, Ji, 0);
    b(this, nc, new Up());
    b(this, ic);
    b(this, sc);
    w(this, ic, n), w(this, sc, s), w(this, ec, a), w(this, wh, l), [t, e] = E(this, Ah, Nm).call(this, t, e);
    const u = w(this, rn, [NaN, NaN, NaN, NaN, t, e]);
    w(this, Bn, [t, e]), w(this, on, [{
      line: u,
      points: o(this, Bn)
    }]), o(this, ir).set(u, 0);
  }
  updateProperty(t, e) {
    t === "stroke-width" && w(this, wh, e);
  }
  isEmpty() {
    return !o(this, on) || o(this, on).length === 0;
  }
  isCancellable() {
    return o(this, Bn).length <= 10;
  }
  add(t, e) {
    [t, e] = E(this, Ah, Nm).call(this, t, e);
    const [n, s, a, l] = o(this, ir).subarray(2, 6), u = t - a, h = e - l;
    return Math.hypot(o(this, ic) * u, o(this, sc) * h) <= 2 ? null : (o(this, Bn).push(t, e), isNaN(n) ? (o(this, ir).set([a, l, t, e], 2), o(this, rn).push(NaN, NaN, NaN, NaN, t, e), {
      path: {
        d: this.toSVGPath()
      }
    }) : (isNaN(o(this, ir)[0]) && o(this, rn).splice(6, 6), o(this, ir).set([n, s, a, l, t, e], 0), o(this, rn).push(...dt.createBezierPoints(n, s, a, l, t, e)), {
      path: {
        d: this.toSVGPath()
      }
    }));
  }
  end(t, e) {
    const n = this.add(t, e);
    return n || (o(this, Bn).length === 2 ? {
      path: {
        d: this.toSVGPath()
      }
    } : null);
  }
  startNew(t, e, n, s, a) {
    w(this, ic, n), w(this, sc, s), w(this, ec, a), [t, e] = E(this, Ah, Nm).call(this, t, e);
    const l = w(this, rn, [NaN, NaN, NaN, NaN, t, e]);
    w(this, Bn, [t, e]);
    const u = o(this, on).at(-1);
    return u && (u.line = new Float32Array(u.line), u.points = new Float32Array(u.points)), o(this, on).push({
      line: l,
      points: o(this, Bn)
    }), o(this, ir).set(l, 0), w(this, Ji, 0), this.toSVGPath(), null;
  }
  getLastElement() {
    return o(this, on).at(-1);
  }
  setLastElement(t) {
    return o(this, on) ? (o(this, on).push(t), w(this, rn, t.line), w(this, Bn, t.points), w(this, Ji, 0), {
      path: {
        d: this.toSVGPath()
      }
    }) : o(this, nc).setLastElement(t);
  }
  removeLastElement() {
    if (!o(this, on))
      return o(this, nc).removeLastElement();
    o(this, on).pop(), w(this, yn, "");
    for (let t = 0, e = o(this, on).length; t < e; t++) {
      const {
        line: n,
        points: s
      } = o(this, on)[t];
      w(this, rn, n), w(this, Bn, s), w(this, Ji, 0), this.toSVGPath();
    }
    return {
      path: {
        d: o(this, yn)
      }
    };
  }
  toSVGPath() {
    const t = dt.svgRound(o(this, rn)[4]), e = dt.svgRound(o(this, rn)[5]);
    if (o(this, Bn).length === 2)
      return w(this, yn, `${o(this, yn)} M ${t} ${e} Z`), o(this, yn);
    if (o(this, Bn).length <= 6) {
      const s = o(this, yn).lastIndexOf("M");
      w(this, yn, `${o(this, yn).slice(0, s)} M ${t} ${e}`), w(this, Ji, 6);
    }
    if (o(this, Bn).length === 4) {
      const s = dt.svgRound(o(this, rn)[10]), a = dt.svgRound(o(this, rn)[11]);
      return w(this, yn, `${o(this, yn)} L ${s} ${a}`), w(this, Ji, 12), o(this, yn);
    }
    const n = [];
    o(this, Ji) === 0 && (n.push(`M ${t} ${e}`), w(this, Ji, 6));
    for (let s = o(this, Ji), a = o(this, rn).length; s < a; s += 6) {
      const [l, u, h, f, g, y] = o(this, rn).slice(s, s + 6).map(dt.svgRound);
      n.push(`C${l} ${u} ${h} ${f} ${g} ${y}`);
    }
    return w(this, yn, o(this, yn) + n.join(" ")), w(this, Ji, o(this, rn).length), o(this, yn);
  }
  getOutlines(t, e, n, s) {
    const a = o(this, on).at(-1);
    return a.line = new Float32Array(a.line), a.points = new Float32Array(a.points), o(this, nc).build(o(this, on), t, e, n, o(this, ec), o(this, wh), s), w(this, ir, null), w(this, rn, null), w(this, on, null), w(this, yn, null), o(this, nc);
  }
  get defaultSVGProperties() {
    return {
      root: {
        viewBox: "0 0 10000 10000"
      },
      rootClass: {
        draw: !0
      },
      bbox: [0, 0, 1, 1]
    };
  }
}
ir = new WeakMap(), rn = new WeakMap(), on = new WeakMap(), ec = new WeakMap(), wh = new WeakMap(), Bn = new WeakMap(), yn = new WeakMap(), Ji = new WeakMap(), nc = new WeakMap(), ic = new WeakMap(), sc = new WeakMap(), Ah = new WeakSet(), Nm = function(t, e) {
  return dt._normalizePoint(t, e, o(this, ic), o(this, sc), o(this, ec));
};
var zn, Mp, Lp, xi, sr, rr, Sh, bh, rc, wn, br, yC, vC, wC;
class Up extends dt {
  constructor() {
    super(...arguments);
    b(this, wn);
    b(this, zn);
    b(this, Mp, 0);
    b(this, Lp);
    b(this, xi);
    b(this, sr);
    b(this, rr);
    b(this, Sh);
    b(this, bh);
    b(this, rc);
  }
  build(e, n, s, a, l, u, h) {
    w(this, sr, n), w(this, rr, s), w(this, Sh, a), w(this, bh, l), w(this, rc, u), w(this, Lp, h ?? 0), w(this, xi, e), E(this, wn, vC).call(this);
  }
  get thickness() {
    return o(this, rc);
  }
  setLastElement(e) {
    return o(this, xi).push(e), {
      path: {
        d: this.toSVGPath()
      }
    };
  }
  removeLastElement() {
    return o(this, xi).pop(), {
      path: {
        d: this.toSVGPath()
      }
    };
  }
  toSVGPath() {
    const e = [];
    for (const {
      line: n
    } of o(this, xi)) {
      if (e.push(`M${dt.svgRound(n[4])} ${dt.svgRound(n[5])}`), n.length === 6) {
        e.push("Z");
        continue;
      }
      if (n.length === 12 && isNaN(n[6])) {
        e.push(`L${dt.svgRound(n[10])} ${dt.svgRound(n[11])}`);
        continue;
      }
      for (let s = 6, a = n.length; s < a; s += 6) {
        const [l, u, h, f, g, y] = n.subarray(s, s + 6).map(dt.svgRound);
        e.push(`C${l} ${u} ${h} ${f} ${g} ${y}`);
      }
    }
    return e.join("");
  }
  serialize([e, n, s, a], l) {
    const u = [], h = [], [f, g, y, v] = E(this, wn, yC).call(this);
    let S, x, _, k, T, R, P, L, D;
    switch (o(this, bh)) {
      case 0:
        D = dt._rescale, S = e, x = n + a, _ = s, k = -a, T = e + f * s, R = n + (1 - g - v) * a, P = e + (f + y) * s, L = n + (1 - g) * a;
        break;
      case 90:
        D = dt._rescaleAndSwap, S = e, x = n, _ = s, k = a, T = e + g * s, R = n + f * a, P = e + (g + v) * s, L = n + (f + y) * a;
        break;
      case 180:
        D = dt._rescale, S = e + s, x = n, _ = -s, k = a, T = e + (1 - f - y) * s, R = n + g * a, P = e + (1 - f) * s, L = n + (g + v) * a;
        break;
      case 270:
        D = dt._rescaleAndSwap, S = e + s, x = n + a, _ = -s, k = -a, T = e + (1 - g - v) * s, R = n + (1 - f - y) * a, P = e + (1 - g) * s, L = n + (1 - f) * a;
        break;
    }
    for (const {
      line: N,
      points: B
    } of o(this, xi))
      u.push(D(N, S, x, _, k, l ? new Array(N.length) : null)), h.push(D(B, S, x, _, k, l ? new Array(B.length) : null));
    return {
      lines: u,
      points: h,
      rect: [T, R, P, L]
    };
  }
  static deserialize(e, n, s, a, l, {
    paths: {
      lines: u,
      points: h
    },
    rotation: f,
    thickness: g
  }) {
    const y = [];
    let v, S, x, _, k;
    switch (f) {
      case 0:
        k = dt._rescale, v = -e / s, S = n / a + 1, x = 1 / s, _ = -1 / a;
        break;
      case 90:
        k = dt._rescaleAndSwap, v = -n / a, S = -e / s, x = 1 / a, _ = 1 / s;
        break;
      case 180:
        k = dt._rescale, v = e / s + 1, S = -n / a, x = -1 / s, _ = 1 / a;
        break;
      case 270:
        k = dt._rescaleAndSwap, v = n / a + 1, S = e / s + 1, x = -1 / a, _ = -1 / s;
        break;
    }
    if (!u) {
      u = [];
      for (const R of h) {
        const P = R.length;
        if (P === 2) {
          u.push(new Float32Array([NaN, NaN, NaN, NaN, R[0], R[1]]));
          continue;
        }
        if (P === 4) {
          u.push(new Float32Array([NaN, NaN, NaN, NaN, R[0], R[1], NaN, NaN, NaN, NaN, R[2], R[3]]));
          continue;
        }
        const L = new Float32Array(3 * (P - 2));
        u.push(L);
        let [D, N, B, $] = R.subarray(0, 4);
        L.set([NaN, NaN, NaN, NaN, D, N], 0);
        for (let V = 4; V < P; V += 2) {
          const X = R[V], q = R[V + 1];
          L.set(dt.createBezierPoints(D, N, B, $, X, q), (V - 2) * 3), [D, N, B, $] = [B, $, X, q];
        }
      }
    }
    for (let R = 0, P = u.length; R < P; R++)
      y.push({
        line: k(u[R].map((L) => L ?? NaN), v, S, x, _),
        points: k(h[R].map((L) => L ?? NaN), v, S, x, _)
      });
    const T = new this.prototype.constructor();
    return T.build(y, s, a, 1, f, g, l), T;
  }
  get box() {
    return o(this, zn);
  }
  updateProperty(e, n) {
    return e === "stroke-width" ? E(this, wn, wC).call(this, n) : null;
  }
  updateParentDimensions([e, n], s) {
    const [a, l] = E(this, wn, br).call(this);
    w(this, sr, e), w(this, rr, n), w(this, Sh, s);
    const [u, h] = E(this, wn, br).call(this), f = u - a, g = h - l, y = o(this, zn);
    return y[0] -= f, y[1] -= g, y[2] += 2 * f, y[3] += 2 * g, y;
  }
  updateRotation(e) {
    return w(this, Mp, e), {
      path: {
        transform: this.rotationTransform
      }
    };
  }
  get viewBox() {
    return o(this, zn).map(dt.svgRound).join(" ");
  }
  get defaultProperties() {
    const [e, n] = o(this, zn);
    return {
      root: {
        viewBox: this.viewBox
      },
      path: {
        "transform-origin": `${dt.svgRound(e)} ${dt.svgRound(n)}`
      }
    };
  }
  get rotationTransform() {
    const [, , e, n] = o(this, zn);
    let s = 0, a = 0, l = 0, u = 0, h = 0, f = 0;
    switch (o(this, Mp)) {
      case 90:
        a = n / e, l = -e / n, h = e;
        break;
      case 180:
        s = -1, u = -1, h = e, f = n;
        break;
      case 270:
        a = -n / e, l = e / n, f = n;
        break;
      default:
        return "";
    }
    return `matrix(${s} ${a} ${l} ${u} ${dt.svgRound(h)} ${dt.svgRound(f)})`;
  }
  getPathResizingSVGProperties([e, n, s, a]) {
    const [l, u] = E(this, wn, br).call(this), [h, f, g, y] = o(this, zn);
    if (Math.abs(g - l) <= dt.PRECISION || Math.abs(y - u) <= dt.PRECISION) {
      const k = e + s / 2 - (h + g / 2), T = n + a / 2 - (f + y / 2);
      return {
        path: {
          "transform-origin": `${dt.svgRound(e)} ${dt.svgRound(n)}`,
          transform: `${this.rotationTransform} translate(${k} ${T})`
        }
      };
    }
    const v = (s - 2 * l) / (g - 2 * l), S = (a - 2 * u) / (y - 2 * u), x = g / s, _ = y / a;
    return {
      path: {
        "transform-origin": `${dt.svgRound(h)} ${dt.svgRound(f)}`,
        transform: `${this.rotationTransform} scale(${x} ${_}) translate(${dt.svgRound(l)} ${dt.svgRound(u)}) scale(${v} ${S}) translate(${dt.svgRound(-l)} ${dt.svgRound(-u)})`
      }
    };
  }
  getPathResizedSVGProperties([e, n, s, a]) {
    const [l, u] = E(this, wn, br).call(this), h = o(this, zn), [f, g, y, v] = h;
    if (h[0] = e, h[1] = n, h[2] = s, h[3] = a, Math.abs(y - l) <= dt.PRECISION || Math.abs(v - u) <= dt.PRECISION) {
      const T = e + s / 2 - (f + y / 2), R = n + a / 2 - (g + v / 2);
      for (const {
        line: P,
        points: L
      } of o(this, xi))
        dt._translate(P, T, R, P), dt._translate(L, T, R, L);
      return {
        root: {
          viewBox: this.viewBox
        },
        path: {
          "transform-origin": `${dt.svgRound(e)} ${dt.svgRound(n)}`,
          transform: this.rotationTransform || null,
          d: this.toSVGPath()
        }
      };
    }
    const S = (s - 2 * l) / (y - 2 * l), x = (a - 2 * u) / (v - 2 * u), _ = -S * (f + l) + e + l, k = -x * (g + u) + n + u;
    if (S !== 1 || x !== 1 || _ !== 0 || k !== 0)
      for (const {
        line: T,
        points: R
      } of o(this, xi))
        dt._rescale(T, _, k, S, x, T), dt._rescale(R, _, k, S, x, R);
    return {
      root: {
        viewBox: this.viewBox
      },
      path: {
        "transform-origin": `${dt.svgRound(e)} ${dt.svgRound(n)}`,
        transform: this.rotationTransform || null,
        d: this.toSVGPath()
      }
    };
  }
  getPathTranslatedSVGProperties([e, n], s) {
    const [a, l] = s, u = o(this, zn), h = e - u[0], f = n - u[1];
    if (o(this, sr) === a && o(this, rr) === l)
      for (const {
        line: g,
        points: y
      } of o(this, xi))
        dt._translate(g, h, f, g), dt._translate(y, h, f, y);
    else {
      const g = o(this, sr) / a, y = o(this, rr) / l;
      w(this, sr, a), w(this, rr, l);
      for (const {
        line: v,
        points: S
      } of o(this, xi))
        dt._rescale(v, h, f, g, y, v), dt._rescale(S, h, f, g, y, S);
      u[2] *= g, u[3] *= y;
    }
    return u[0] = e, u[1] = n, {
      root: {
        viewBox: this.viewBox
      },
      path: {
        d: this.toSVGPath(),
        "transform-origin": `${dt.svgRound(e)} ${dt.svgRound(n)}`
      }
    };
  }
  get defaultSVGProperties() {
    const e = o(this, zn);
    return {
      root: {
        viewBox: this.viewBox
      },
      rootClass: {
        draw: !0
      },
      path: {
        d: this.toSVGPath(),
        "transform-origin": `${dt.svgRound(e[0])} ${dt.svgRound(e[1])}`,
        transform: this.rotationTransform || null
      },
      bbox: e
    };
  }
}
zn = new WeakMap(), Mp = new WeakMap(), Lp = new WeakMap(), xi = new WeakMap(), sr = new WeakMap(), rr = new WeakMap(), Sh = new WeakMap(), bh = new WeakMap(), rc = new WeakMap(), wn = new WeakSet(), br = function(e = o(this, rc)) {
  const n = o(this, Lp) + e / 2 * o(this, Sh);
  return o(this, bh) % 180 === 0 ? [n / o(this, sr), n / o(this, rr)] : [n / o(this, rr), n / o(this, sr)];
}, yC = function() {
  const [e, n, s, a] = o(this, zn), [l, u] = E(this, wn, br).call(this, 0);
  return [e + l, n + u, s - 2 * l, a - 2 * u];
}, vC = function() {
  const e = w(this, zn, new Float32Array([1 / 0, 1 / 0, -1 / 0, -1 / 0]));
  for (const {
    line: a
  } of o(this, xi)) {
    if (a.length <= 12) {
      for (let h = 4, f = a.length; h < f; h += 6)
        ft.pointBoundingBox(a[h], a[h + 1], e);
      continue;
    }
    let l = a[4], u = a[5];
    for (let h = 6, f = a.length; h < f; h += 6) {
      const [g, y, v, S, x, _] = a.subarray(h, h + 6);
      ft.bezierBoundingBox(l, u, g, y, v, S, x, _, e), l = x, u = _;
    }
  }
  const [n, s] = E(this, wn, br).call(this);
  e[0] = Ln(e[0] - n, 0, 1), e[1] = Ln(e[1] - s, 0, 1), e[2] = Ln(e[2] + n, 0, 1), e[3] = Ln(e[3] + s, 0, 1), e[2] -= e[0], e[3] -= e[1];
}, wC = function(e) {
  const [n, s] = E(this, wn, br).call(this);
  w(this, rc, e);
  const [a, l] = E(this, wn, br).call(this), [u, h] = [a - n, l - s], f = o(this, zn);
  return f[0] -= u, f[1] -= h, f[2] += 2 * u, f[3] += 2 * h, f;
};
class Cy extends gC {
  constructor(t) {
    super(), this._viewParameters = t, super.updateProperties({
      fill: "none",
      stroke: fe._defaultLineColor,
      "stroke-opacity": 1,
      "stroke-width": 1,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "stroke-miterlimit": 10
    });
  }
  updateSVGProperty(t, e) {
    t === "stroke-width" && (e ?? (e = this["stroke-width"]), e *= this._viewParameters.realScale), super.updateSVGProperty(t, e);
  }
  clone() {
    const t = new Cy(this._viewParameters);
    return t.updateAll(this), t;
  }
}
var vy, AC;
const Jc = class Jc extends Zm {
  constructor(e) {
    super({
      ...e,
      name: "inkEditor"
    });
    b(this, vy);
    this._willKeepAspectRatio = !0, this.defaultL10nId = "pdfjs-editor-ink-editor";
  }
  static initialize(e, n) {
    fe.initialize(e, n), this._defaultDrawingOptions = new Cy(n.viewParameters);
  }
  static getDefaultDrawingOptions(e) {
    const n = this._defaultDrawingOptions.clone();
    return n.updateProperties(e), n;
  }
  static get supportMultipleDrawings() {
    return !0;
  }
  static get typesMap() {
    return It(this, "typesMap", /* @__PURE__ */ new Map([[jt.INK_THICKNESS, "stroke-width"], [jt.INK_COLOR, "stroke"], [jt.INK_OPACITY, "stroke-opacity"]]));
  }
  static createDrawerInstance(e, n, s, a, l) {
    return new GP(e, n, s, a, l, this._defaultDrawingOptions["stroke-width"]);
  }
  static deserializeDraw(e, n, s, a, l, u) {
    return Up.deserialize(e, n, s, a, l, u);
  }
  static async deserialize(e, n, s) {
    let a = null;
    if (e instanceof AA) {
      const {
        data: {
          inkLists: u,
          rect: h,
          rotation: f,
          id: g,
          color: y,
          opacity: v,
          borderStyle: {
            rawWidth: S
          },
          popupRef: x,
          richText: _,
          contentsObj: k,
          creationDate: T,
          modificationDate: R
        },
        parent: {
          page: {
            pageNumber: P
          }
        }
      } = e;
      a = e = {
        annotationType: Pt.INK,
        color: Array.from(y),
        thickness: S,
        opacity: v,
        paths: {
          points: u
        },
        boxes: null,
        pageIndex: P - 1,
        rect: h.slice(0),
        rotation: f,
        annotationElementId: g,
        id: g,
        deleted: !1,
        popupRef: x,
        richText: _,
        comment: (k == null ? void 0 : k.str) || null,
        creationDate: T,
        modificationDate: R
      };
    }
    const l = await super.deserialize(e, n, s);
    return l._initialData = a, e.comment && l.setCommentData(e), l;
  }
  get toolbarButtons() {
    return this._colorPicker || (this._colorPicker = new Ym(this)), [["colorPicker", this._colorPicker]];
  }
  get colorType() {
    return jt.INK_COLOR;
  }
  get color() {
    return this._drawingOptions.stroke;
  }
  get opacity() {
    return this._drawingOptions["stroke-opacity"];
  }
  onScaleChanging() {
    if (!this.parent)
      return;
    super.onScaleChanging();
    const {
      _drawId: e,
      _drawingOptions: n,
      parent: s
    } = this;
    n.updateSVGProperty("stroke-width"), s.drawLayer.updateProperties(e, n.toSVGProperties());
  }
  static onScaleChangingWhenDrawing() {
    const e = this._currentParent;
    e && (super.onScaleChangingWhenDrawing(), this._defaultDrawingOptions.updateSVGProperty("stroke-width"), e.drawLayer.updateProperties(this._currentDrawId, this._defaultDrawingOptions.toSVGProperties()));
  }
  createDrawingOptions({
    color: e,
    thickness: n,
    opacity: s
  }) {
    this._drawingOptions = Jc.getDefaultDrawingOptions({
      stroke: ft.makeHexColor(...e),
      "stroke-width": n,
      "stroke-opacity": s
    });
  }
  serialize(e = !1) {
    if (this.isEmpty())
      return null;
    if (this.deleted)
      return this.serializeDeleted();
    const {
      lines: n,
      points: s
    } = this.serializeDraw(e), {
      _drawingOptions: {
        stroke: a,
        "stroke-opacity": l,
        "stroke-width": u
      }
    } = this, h = Object.assign(super.serialize(e), {
      color: fe._colorManager.convert(a),
      opacity: l,
      thickness: u,
      paths: {
        lines: n,
        points: s
      }
    });
    return this.addComment(h), e ? (h.isCopy = !0, h) : this.annotationElementId && !E(this, vy, AC).call(this, h) ? null : (h.id = this.annotationElementId, h);
  }
  renderAnnotationElement(e) {
    if (this.deleted)
      return e.hide(), null;
    const {
      points: n,
      rect: s
    } = this.serializeDraw(!1);
    return e.updateEdited({
      rect: s,
      thickness: this._drawingOptions["stroke-width"],
      points: n,
      popup: this.comment
    }), null;
  }
};
vy = new WeakSet(), AC = function(e) {
  const {
    color: n,
    thickness: s,
    opacity: a,
    pageIndex: l
  } = this._initialData;
  return this.hasEditedComment || this._hasBeenMoved || this._hasBeenResized || e.color.some((u, h) => u !== n[h]) || e.thickness !== s || e.opacity !== a || e.pageIndex !== l;
}, Z(Jc, "_type", "ink"), Z(Jc, "_editorType", Pt.INK), Z(Jc, "_defaultDrawingOptions", null);
let Ww = Jc;
class Gw extends Up {
  toSVGPath() {
    let t = super.toSVGPath();
    return t.endsWith("Z") || (t += "Z"), t;
  }
}
const cm = 8, bd = 3;
var oc, Yt, Xw, _s, SC, bC, qw, Om, EC, _C, CC, Yw, Kw, xC;
class cr {
  static extractContoursFromText(t, {
    fontFamily: e,
    fontStyle: n,
    fontWeight: s
  }, a, l, u, h) {
    let f = new OffscreenCanvas(1, 1), g = f.getContext("2d", {
      alpha: !1
    });
    const y = 200, v = g.font = `${n} ${s} ${y}px ${e}`, {
      actualBoundingBoxLeft: S,
      actualBoundingBoxRight: x,
      actualBoundingBoxAscent: _,
      actualBoundingBoxDescent: k,
      fontBoundingBoxAscent: T,
      fontBoundingBoxDescent: R,
      width: P
    } = g.measureText(t), L = 1.5, D = Math.ceil(Math.max(Math.abs(S) + Math.abs(x) || 0, P) * L), N = Math.ceil(Math.max(Math.abs(_) + Math.abs(k) || y, Math.abs(T) + Math.abs(R) || y) * L);
    f = new OffscreenCanvas(D, N), g = f.getContext("2d", {
      alpha: !0,
      willReadFrequently: !0
    }), g.font = v, g.filter = "grayscale(1)", g.fillStyle = "white", g.fillRect(0, 0, D, N), g.fillStyle = "black", g.fillText(t, D * (L - 1) / 2, N * (3 - L) / 2);
    const B = E(this, Yt, Yw).call(this, g.getImageData(0, 0, D, N).data), $ = E(this, Yt, CC).call(this, B), V = E(this, Yt, Kw).call(this, $), X = E(this, Yt, qw).call(this, B, D, N, V);
    return this.processDrawnLines({
      lines: {
        curves: X,
        width: D,
        height: N
      },
      pageWidth: a,
      pageHeight: l,
      rotation: u,
      innerMargin: h,
      mustSmooth: !0,
      areContours: !0
    });
  }
  static process(t, e, n, s, a) {
    const [l, u, h] = E(this, Yt, xC).call(this, t), [f, g] = E(this, Yt, _C).call(this, l, u, h, Math.hypot(u, h) * o(this, oc).sigmaSFactor, o(this, oc).sigmaR, o(this, oc).kernelSize), y = E(this, Yt, Kw).call(this, g), v = E(this, Yt, qw).call(this, f, u, h, y);
    return this.processDrawnLines({
      lines: {
        curves: v,
        width: u,
        height: h
      },
      pageWidth: e,
      pageHeight: n,
      rotation: s,
      innerMargin: a,
      mustSmooth: !0,
      areContours: !0
    });
  }
  static processDrawnLines({
    lines: t,
    pageWidth: e,
    pageHeight: n,
    rotation: s,
    innerMargin: a,
    mustSmooth: l,
    areContours: u
  }) {
    s % 180 !== 0 && ([e, n] = [n, e]);
    const {
      curves: h,
      width: f,
      height: g
    } = t, y = t.thickness ?? 0, v = [], S = Math.min(e / f, n / g), x = S / e, _ = S / n, k = [];
    for (const {
      points: R
    } of h) {
      const P = l ? E(this, Yt, EC).call(this, R) : R;
      if (!P)
        continue;
      k.push(P);
      const L = P.length, D = new Float32Array(L), N = new Float32Array(3 * (L === 2 ? 2 : L - 2));
      if (v.push({
        line: N,
        points: D
      }), L === 2) {
        D[0] = P[0] * x, D[1] = P[1] * _, N.set([NaN, NaN, NaN, NaN, D[0], D[1]], 0);
        continue;
      }
      let [B, $, V, X] = P;
      B *= x, $ *= _, V *= x, X *= _, D.set([B, $, V, X], 0), N.set([NaN, NaN, NaN, NaN, B, $], 0);
      for (let q = 4; q < L; q += 2) {
        const it = D[q] = P[q] * x, lt = D[q + 1] = P[q + 1] * _;
        N.set(dt.createBezierPoints(B, $, V, X, it, lt), (q - 2) * 3), [B, $, V, X] = [V, X, it, lt];
      }
    }
    if (v.length === 0)
      return null;
    const T = u ? new Gw() : new Up();
    return T.build(v, e, n, 1, s, u ? 0 : y, a), {
      outline: T,
      newCurves: k,
      areContours: u,
      thickness: y,
      width: f,
      height: g
    };
  }
  static async compressSignature({
    outlines: t,
    areContours: e,
    thickness: n,
    width: s,
    height: a
  }) {
    let l = 1 / 0, u = -1 / 0, h = 0;
    for (const P of t) {
      h += P.length;
      for (let L = 2, D = P.length; L < D; L++) {
        const N = P[L] - P[L - 2];
        l = Math.min(l, N), u = Math.max(u, N);
      }
    }
    let f;
    l >= -128 && u <= 127 ? f = Int8Array : l >= -32768 && u <= 32767 ? f = Int16Array : f = Int32Array;
    const g = t.length, y = cm + bd * g, v = new Uint32Array(y);
    let S = 0;
    v[S++] = y * Uint32Array.BYTES_PER_ELEMENT + (h - 2 * g) * f.BYTES_PER_ELEMENT, v[S++] = 0, v[S++] = s, v[S++] = a, v[S++] = e ? 0 : 1, v[S++] = Math.max(0, Math.floor(n ?? 0)), v[S++] = g, v[S++] = f.BYTES_PER_ELEMENT;
    for (const P of t)
      v[S++] = P.length - 2, v[S++] = P[0], v[S++] = P[1];
    const x = new CompressionStream("deflate-raw"), _ = x.writable.getWriter();
    await _.ready, _.write(v);
    const k = f.prototype.constructor;
    for (const P of t) {
      const L = new k(P.length - 2);
      for (let D = 2, N = P.length; D < N; D++)
        L[D - 2] = P[D] - P[D - 2];
      _.write(L);
    }
    _.close();
    const T = await new Response(x.readable).arrayBuffer(), R = new Uint8Array(T);
    return b1(R);
  }
  static async decompressSignature(t) {
    try {
      const e = yk(t), {
        readable: n,
        writable: s
      } = new DecompressionStream("deflate-raw"), a = s.getWriter();
      await a.ready, a.write(e).then(async () => {
        await a.ready, await a.close();
      }).catch(() => {
      });
      let l = null, u = 0;
      for await (const P of n)
        l || (l = new Uint8Array(new Uint32Array(P.buffer, 0, 4)[0])), l.set(P, u), u += P.length;
      const h = new Uint32Array(l.buffer, 0, l.length >> 2), f = h[1];
      if (f !== 0)
        throw new Error(`Invalid version: ${f}`);
      const g = h[2], y = h[3], v = h[4] === 0, S = h[5], x = h[6], _ = h[7], k = [], T = (cm + bd * x) * Uint32Array.BYTES_PER_ELEMENT;
      let R;
      switch (_) {
        case Int8Array.BYTES_PER_ELEMENT:
          R = new Int8Array(l.buffer, T);
          break;
        case Int16Array.BYTES_PER_ELEMENT:
          R = new Int16Array(l.buffer, T);
          break;
        case Int32Array.BYTES_PER_ELEMENT:
          R = new Int32Array(l.buffer, T);
          break;
      }
      u = 0;
      for (let P = 0; P < x; P++) {
        const L = h[bd * P + cm], D = new Float32Array(L + 2);
        k.push(D);
        for (let N = 0; N < bd - 1; N++)
          D[N] = h[bd * P + cm + N + 1];
        for (let N = 0; N < L; N++)
          D[N + 2] = D[N] + R[u++];
      }
      return {
        areContours: v,
        thickness: S,
        outlines: k,
        width: g,
        height: y
      };
    } catch (e) {
      return Rt(`decompressSignature: ${e}`), null;
    }
  }
}
oc = new WeakMap(), Yt = new WeakSet(), Xw = function(t, e, n, s) {
  return n -= t, s -= e, n === 0 ? s > 0 ? 0 : 4 : n === 1 ? s + 6 : 2 - s;
}, _s = new WeakMap(), SC = function(t, e, n, s, a, l, u) {
  const h = E(this, Yt, Xw).call(this, n, s, a, l);
  for (let f = 0; f < 8; f++) {
    const g = (-f + h - u + 16) % 8, y = o(this, _s)[2 * g], v = o(this, _s)[2 * g + 1];
    if (t[(n + y) * e + (s + v)] !== 0)
      return g;
  }
  return -1;
}, bC = function(t, e, n, s, a, l, u) {
  const h = E(this, Yt, Xw).call(this, n, s, a, l);
  for (let f = 0; f < 8; f++) {
    const g = (f + h + u + 16) % 8, y = o(this, _s)[2 * g], v = o(this, _s)[2 * g + 1];
    if (t[(n + y) * e + (s + v)] !== 0)
      return g;
  }
  return -1;
}, qw = function(t, e, n, s) {
  const a = t.length, l = new Int32Array(a);
  for (let g = 0; g < a; g++)
    l[g] = t[g] <= s ? 1 : 0;
  for (let g = 1; g < n - 1; g++)
    l[g * e] = l[g * e + e - 1] = 0;
  for (let g = 0; g < e; g++)
    l[g] = l[e * n - 1 - g] = 0;
  let u = 1, h;
  const f = [];
  for (let g = 1; g < n - 1; g++) {
    h = 1;
    for (let y = 1; y < e - 1; y++) {
      const v = g * e + y, S = l[v];
      if (S === 0)
        continue;
      let x = g, _ = y;
      if (S === 1 && l[v - 1] === 0)
        u += 1, _ -= 1;
      else if (S >= 1 && l[v + 1] === 0)
        u += 1, _ += 1, S > 1 && (h = S);
      else {
        S !== 1 && (h = Math.abs(S));
        continue;
      }
      const k = [y, g], T = _ === y + 1, R = {
        isHole: T,
        points: k,
        id: u,
        parent: 0
      };
      f.push(R);
      let P;
      for (const q of f)
        if (q.id === h) {
          P = q;
          break;
        }
      P ? P.isHole ? R.parent = T ? P.parent : h : R.parent = T ? h : P.parent : R.parent = T ? h : 0;
      const L = E(this, Yt, SC).call(this, l, e, g, y, x, _, 0);
      if (L === -1) {
        l[v] = -u, l[v] !== 1 && (h = Math.abs(l[v]));
        continue;
      }
      let D = o(this, _s)[2 * L], N = o(this, _s)[2 * L + 1];
      const B = g + D, $ = y + N;
      x = B, _ = $;
      let V = g, X = y;
      for (; ; ) {
        const q = E(this, Yt, bC).call(this, l, e, V, X, x, _, 1);
        D = o(this, _s)[2 * q], N = o(this, _s)[2 * q + 1];
        const it = V + D, lt = X + N;
        k.push(lt, it);
        const rt = V * e + X;
        if (l[rt + 1] === 0 ? l[rt] = -u : l[rt] === 1 && (l[rt] = u), it === g && lt === y && V === B && X === $) {
          l[v] !== 1 && (h = Math.abs(l[v]));
          break;
        } else
          x = V, _ = X, V = it, X = lt;
      }
    }
  }
  return f;
}, Om = function(t, e, n, s) {
  if (n - e <= 4) {
    for (let B = e; B < n - 2; B += 2)
      s.push(t[B], t[B + 1]);
    return;
  }
  const a = t[e], l = t[e + 1], u = t[n - 4] - a, h = t[n - 3] - l, f = Math.hypot(u, h), g = u / f, y = h / f, v = g * l - y * a, S = h / u, x = 1 / f, _ = Math.atan(S), k = Math.cos(_), T = Math.sin(_), R = x * (Math.abs(k) + Math.abs(T)), P = x * (1 - R + R ** 2), L = Math.max(Math.atan(Math.abs(T + k) * P), Math.atan(Math.abs(T - k) * P));
  let D = 0, N = e;
  for (let B = e + 2; B < n - 2; B += 2) {
    const $ = Math.abs(v - g * t[B + 1] + y * t[B]);
    $ > D && (N = B, D = $);
  }
  D > (f * L) ** 2 ? (E(this, Yt, Om).call(this, t, e, N + 2, s), E(this, Yt, Om).call(this, t, N, n, s)) : s.push(a, l);
}, EC = function(t) {
  const e = [], n = t.length;
  return E(this, Yt, Om).call(this, t, 0, n, e), e.push(t[n - 2], t[n - 1]), e.length <= 4 ? null : e;
}, _C = function(t, e, n, s, a, l) {
  const u = new Float32Array(l ** 2), h = -2 * s ** 2, f = l >> 1;
  for (let _ = 0; _ < l; _++) {
    const k = (_ - f) ** 2;
    for (let T = 0; T < l; T++)
      u[_ * l + T] = Math.exp((k + (T - f) ** 2) / h);
  }
  const g = new Float32Array(256), y = -2 * a ** 2;
  for (let _ = 0; _ < 256; _++)
    g[_] = Math.exp(_ ** 2 / y);
  const v = t.length, S = new Uint8Array(v), x = new Uint32Array(256);
  for (let _ = 0; _ < n; _++)
    for (let k = 0; k < e; k++) {
      const T = _ * e + k, R = t[T];
      let P = 0, L = 0;
      for (let N = 0; N < l; N++) {
        const B = _ + N - f;
        if (!(B < 0 || B >= n))
          for (let $ = 0; $ < l; $++) {
            const V = k + $ - f;
            if (V < 0 || V >= e)
              continue;
            const X = t[B * e + V], q = u[N * l + $] * g[Math.abs(X - R)];
            P += X * q, L += q;
          }
      }
      const D = S[T] = Math.round(P / L);
      x[D]++;
    }
  return [S, x];
}, CC = function(t) {
  const e = new Uint32Array(256);
  for (const n of t)
    e[n]++;
  return e;
}, Yw = function(t) {
  const e = t.length, n = new Uint8ClampedArray(e >> 2);
  let s = -1 / 0, a = 1 / 0;
  for (let u = 0, h = n.length; u < h; u++) {
    const f = n[u] = t[u << 2];
    s = Math.max(s, f), a = Math.min(a, f);
  }
  const l = 255 / (s - a);
  for (let u = 0, h = n.length; u < h; u++)
    n[u] = (n[u] - a) * l;
  return n;
}, Kw = function(t) {
  let e, n = -1 / 0, s = -1 / 0;
  const a = t.findIndex((h) => h !== 0);
  let l = a, u = a;
  for (e = a; e < 256; e++) {
    const h = t[e];
    h > n && (e - l > s && (s = e - l, u = e - 1), n = h, l = e);
  }
  for (e = u - 1; e >= 0 && !(t[e] > t[e + 1]); e--)
    ;
  return e;
}, xC = function(t) {
  const e = t, {
    width: n,
    height: s
  } = t, {
    maxDim: a
  } = o(this, oc);
  let l = n, u = s;
  if (n > a || s > a) {
    let v = n, S = s, x = Math.log2(Math.max(n, s) / a);
    const _ = Math.floor(x);
    x = x === _ ? _ - 1 : _;
    for (let T = 0; T < x; T++) {
      l = Math.ceil(v / 2), u = Math.ceil(S / 2);
      const R = new OffscreenCanvas(l, u);
      R.getContext("2d").drawImage(t, 0, 0, v, S, 0, 0, l, u), v = l, S = u, t !== e && t.close(), t = R.transferToImageBitmap();
    }
    const k = Math.min(a / l, a / u);
    l = Math.round(l * k), u = Math.round(u * k);
  }
  const f = new OffscreenCanvas(l, u).getContext("2d", {
    willReadFrequently: !0
  });
  f.fillStyle = "white", f.fillRect(0, 0, l, u), f.filter = "grayscale(1)", f.drawImage(t, 0, 0, t.width, t.height, 0, 0, l, u);
  const g = f.getImageData(0, 0, l, u).data;
  return [E(this, Yt, Yw).call(this, g), l, u];
}, b(cr, Yt), b(cr, oc, {
  maxDim: 512,
  sigmaSFactor: 0.02,
  sigmaR: 25,
  kernelSize: 16
}), b(cr, _s, new Int32Array([0, 1, -1, 1, -1, 0, -1, -1, 0, -1, 1, -1, 1, 0, 1, 1]));
class bA extends gC {
  constructor() {
    super(), super.updateProperties({
      fill: fe._defaultLineColor,
      "stroke-width": 0
    });
  }
  clone() {
    const t = new bA();
    return t.updateAll(this), t;
  }
}
class EA extends Cy {
  constructor(t) {
    super(t), super.updateProperties({
      stroke: fe._defaultLineColor,
      "stroke-width": 1
    });
  }
  clone() {
    const t = new EA(this._viewParameters);
    return t.updateAll(this), t;
  }
}
var va, or, wa, ac;
const ui = class ui extends Zm {
  constructor(e) {
    super({
      ...e,
      mustBeCommitted: !0,
      name: "signatureEditor"
    });
    b(this, va, !1);
    b(this, or, null);
    b(this, wa, null);
    b(this, ac, null);
    this._willKeepAspectRatio = !0, w(this, wa, e.signatureData || null), w(this, or, null), this.defaultL10nId = "pdfjs-editor-signature-editor1";
  }
  static initialize(e, n) {
    fe.initialize(e, n), this._defaultDrawingOptions = new bA(), this._defaultDrawnSignatureOptions = new EA(n.viewParameters);
  }
  static getDefaultDrawingOptions(e) {
    const n = this._defaultDrawingOptions.clone();
    return n.updateProperties(e), n;
  }
  static get supportMultipleDrawings() {
    return !1;
  }
  static get typesMap() {
    return It(this, "typesMap", /* @__PURE__ */ new Map());
  }
  static get isDrawer() {
    return !1;
  }
  get telemetryFinalData() {
    return {
      type: "signature",
      hasDescription: !!o(this, or)
    };
  }
  static computeTelemetryFinalData(e) {
    const n = e.get("hasDescription");
    return {
      hasAltText: n.get(!0) ?? 0,
      hasNoAltText: n.get(!1) ?? 0
    };
  }
  get isResizable() {
    return !0;
  }
  onScaleChanging() {
    this._drawId !== null && super.onScaleChanging();
  }
  render() {
    if (this.div)
      return this.div;
    let e, n;
    const {
      _isCopy: s
    } = this;
    if (s && (this._isCopy = !1, e = this.x, n = this.y), super.render(), this._drawId === null)
      if (o(this, wa)) {
        const {
          lines: a,
          mustSmooth: l,
          areContours: u,
          description: h,
          uuid: f,
          heightInPage: g
        } = o(this, wa), {
          rawDims: {
            pageWidth: y,
            pageHeight: v
          },
          rotation: S
        } = this.parent.viewport, x = cr.processDrawnLines({
          lines: a,
          pageWidth: y,
          pageHeight: v,
          rotation: S,
          innerMargin: ui._INNER_MARGIN,
          mustSmooth: l,
          areContours: u
        });
        this.addSignature(x, g, h, f);
      } else
        this.div.setAttribute("data-l10n-args", JSON.stringify({
          description: ""
        })), this.div.hidden = !0, this._uiManager.getSignature(this);
    else
      this.div.setAttribute("data-l10n-args", JSON.stringify({
        description: o(this, or) || ""
      }));
    return s && (this._isCopy = !0, this._moveAfterPaste(e, n)), this.div;
  }
  setUuid(e) {
    w(this, ac, e), this.addEditToolbar();
  }
  getUuid() {
    return o(this, ac);
  }
  get description() {
    return o(this, or);
  }
  set description(e) {
    w(this, or, e), this.div && (this.div.setAttribute("data-l10n-args", JSON.stringify({
      description: e
    })), super.addEditToolbar().then((n) => {
      n == null || n.updateEditSignatureButton(e);
    }));
  }
  getSignaturePreview() {
    const {
      newCurves: e,
      areContours: n,
      thickness: s,
      width: a,
      height: l
    } = o(this, wa), u = Math.max(a, l), h = cr.processDrawnLines({
      lines: {
        curves: e.map((f) => ({
          points: f
        })),
        thickness: s,
        width: a,
        height: l
      },
      pageWidth: u,
      pageHeight: u,
      rotation: 0,
      innerMargin: 0,
      mustSmooth: !1,
      areContours: n
    });
    return {
      areContours: n,
      outline: h.outline
    };
  }
  get toolbarButtons() {
    return this._uiManager.signatureManager ? [["editSignature", this._uiManager.signatureManager]] : super.toolbarButtons;
  }
  addSignature(e, n, s, a) {
    const {
      x: l,
      y: u
    } = this, {
      outline: h
    } = w(this, wa, e);
    w(this, va, h instanceof Gw), this.description = s;
    let f;
    o(this, va) ? f = ui.getDefaultDrawingOptions() : (f = ui._defaultDrawnSignatureOptions.clone(), f.updateProperties({
      "stroke-width": h.thickness
    })), this._addOutlines({
      drawOutlines: h,
      drawingOptions: f
    });
    const [, g] = this.pageDimensions;
    let y = n / g;
    y = y >= 1 ? 0.5 : y, this.width *= y / this.height, this.width >= 1 && (y *= 0.9 / this.width, this.width = 0.9), this.height = y, this.setDims(), this.x = l, this.y = u, this.center(), this._onResized(), this.onScaleChanging(), this.rotate(), this._uiManager.addToAnnotationStorage(this), this.setUuid(a), this._reportTelemetry({
      action: "pdfjs.signature.inserted",
      data: {
        hasBeenSaved: !!a,
        hasDescription: !!s
      }
    }), this.div.hidden = !1;
  }
  getFromImage(e) {
    const {
      rawDims: {
        pageWidth: n,
        pageHeight: s
      },
      rotation: a
    } = this.parent.viewport;
    return cr.process(e, n, s, a, ui._INNER_MARGIN);
  }
  getFromText(e, n) {
    const {
      rawDims: {
        pageWidth: s,
        pageHeight: a
      },
      rotation: l
    } = this.parent.viewport;
    return cr.extractContoursFromText(e, n, s, a, l, ui._INNER_MARGIN);
  }
  getDrawnSignature(e) {
    const {
      rawDims: {
        pageWidth: n,
        pageHeight: s
      },
      rotation: a
    } = this.parent.viewport;
    return cr.processDrawnLines({
      lines: e,
      pageWidth: n,
      pageHeight: s,
      rotation: a,
      innerMargin: ui._INNER_MARGIN,
      mustSmooth: !1,
      areContours: !1
    });
  }
  createDrawingOptions({
    areContours: e,
    thickness: n
  }) {
    e ? this._drawingOptions = ui.getDefaultDrawingOptions() : (this._drawingOptions = ui._defaultDrawnSignatureOptions.clone(), this._drawingOptions.updateProperties({
      "stroke-width": n
    }));
  }
  serialize(e = !1) {
    if (this.isEmpty())
      return null;
    const {
      lines: n,
      points: s
    } = this.serializeDraw(e), {
      _drawingOptions: {
        "stroke-width": a
      }
    } = this, l = Object.assign(super.serialize(e), {
      isSignature: !0,
      areContours: o(this, va),
      color: [0, 0, 0],
      thickness: o(this, va) ? 0 : a
    });
    return this.addComment(l), e ? (l.paths = {
      lines: n,
      points: s
    }, l.uuid = o(this, ac), l.isCopy = !0) : l.lines = n, o(this, or) && (l.accessibilityData = {
      type: "Figure",
      alt: o(this, or)
    }), l;
  }
  static deserializeDraw(e, n, s, a, l, u) {
    return u.areContours ? Gw.deserialize(e, n, s, a, l, u) : Up.deserialize(e, n, s, a, l, u);
  }
  static async deserialize(e, n, s) {
    var l;
    const a = await super.deserialize(e, n, s);
    return w(a, va, e.areContours), a.description = ((l = e.accessibilityData) == null ? void 0 : l.alt) || "", w(a, ac, e.uuid), a;
  }
};
va = new WeakMap(), or = new WeakMap(), wa = new WeakMap(), ac = new WeakMap(), Z(ui, "_type", "signature"), Z(ui, "_editorType", Pt.SIGNATURE), Z(ui, "_defaultDrawingOptions", null);
let Qw = ui;
var Ae, an, Aa, to, Sa, Eh, eo, lc, ar, Ti, _h, Gt, jd, $d, Bm, zm, Um, Jw, Hm, TC;
class Zw extends fe {
  constructor(e) {
    super({
      ...e,
      name: "stampEditor"
    });
    b(this, Gt);
    b(this, Ae, null);
    b(this, an, null);
    b(this, Aa, null);
    b(this, to, null);
    b(this, Sa, null);
    b(this, Eh, "");
    b(this, eo, null);
    b(this, lc, !1);
    b(this, ar, null);
    b(this, Ti, !1);
    b(this, _h, !1);
    w(this, to, e.bitmapUrl), w(this, Sa, e.bitmapFile), this.defaultL10nId = "pdfjs-editor-stamp-editor";
  }
  static initialize(e, n) {
    fe.initialize(e, n);
  }
  static isHandlingMimeForPasting(e) {
    return Vm.includes(e);
  }
  static paste(e, n) {
    n.pasteEditor({
      mode: Pt.STAMP
    }, {
      bitmapFile: e.getAsFile()
    });
  }
  altTextFinish() {
    this._uiManager.useNewAltTextFlow && (this.div.hidden = !1), super.altTextFinish();
  }
  get telemetryFinalData() {
    var e;
    return {
      type: "stamp",
      hasAltText: !!((e = this.altTextData) != null && e.altText)
    };
  }
  static computeTelemetryFinalData(e) {
    const n = e.get("hasAltText");
    return {
      hasAltText: n.get(!0) ?? 0,
      hasNoAltText: n.get(!1) ?? 0
    };
  }
  async mlGuessAltText(e = null, n = !0) {
    if (this.hasAltTextData())
      return null;
    const {
      mlManager: s
    } = this._uiManager;
    if (!s)
      throw new Error("No ML.");
    if (!await s.isEnabledFor("altText"))
      throw new Error("ML isn't enabled for alt text.");
    const {
      data: a,
      width: l,
      height: u
    } = e || this.copyCanvas(null, null, !0).imageData, h = await s.guess({
      name: "altText",
      request: {
        data: a,
        width: l,
        height: u,
        channels: a.length / (l * u)
      }
    });
    if (!h)
      throw new Error("No response from the AI service.");
    if (h.error)
      throw new Error("Error from the AI service.");
    if (h.cancel)
      return null;
    if (!h.output)
      throw new Error("No valid response from the AI service.");
    const f = h.output;
    return await this.setGuessedAltText(f), n && !this.hasAltTextData() && (this.altTextData = {
      alt: f,
      decorative: !1
    }), f;
  }
  remove() {
    var e;
    o(this, an) && (w(this, Ae, null), this._uiManager.imageManager.deleteId(o(this, an)), (e = o(this, eo)) == null || e.remove(), w(this, eo, null), o(this, ar) && (clearTimeout(o(this, ar)), w(this, ar, null))), super.remove();
  }
  rebuild() {
    if (!this.parent) {
      o(this, an) && E(this, Gt, Bm).call(this);
      return;
    }
    super.rebuild(), this.div !== null && (o(this, an) && o(this, eo) === null && E(this, Gt, Bm).call(this), this.isAttachedToDOM || this.parent.add(this));
  }
  onceAdded(e) {
    this._isDraggable = !0, e && this.div.focus();
  }
  isEmpty() {
    return !(o(this, Aa) || o(this, Ae) || o(this, to) || o(this, Sa) || o(this, an) || o(this, lc));
  }
  get toolbarButtons() {
    return [["altText", this.createAltText()]];
  }
  get isResizable() {
    return !0;
  }
  render() {
    if (this.div)
      return this.div;
    let e, n;
    return this._isCopy && (e = this.x, n = this.y), super.render(), this.div.hidden = !0, this.createAltText(), o(this, lc) || (o(this, Ae) ? E(this, Gt, zm).call(this) : E(this, Gt, Bm).call(this)), this._isCopy && this._moveAfterPaste(e, n), this._uiManager.addShouldRescale(this), this.div;
  }
  setCanvas(e, n) {
    const {
      id: s,
      bitmap: a
    } = this._uiManager.imageManager.getFromCanvas(e, n);
    n.remove(), s && this._uiManager.imageManager.isValidId(s) && (w(this, an, s), a && w(this, Ae, a), w(this, lc, !1), E(this, Gt, zm).call(this));
  }
  _onResized() {
    this.onScaleChanging();
  }
  onScaleChanging() {
    if (!this.parent)
      return;
    o(this, ar) !== null && clearTimeout(o(this, ar)), w(this, ar, setTimeout(() => {
      w(this, ar, null), E(this, Gt, Jw).call(this);
    }, 200));
  }
  copyCanvas(e, n, s = !1) {
    e || (e = 224);
    const {
      width: a,
      height: l
    } = o(this, Ae), u = new ks();
    let h = o(this, Ae), f = a, g = l, y = null;
    if (n) {
      if (a > n || l > n) {
        const B = Math.min(n / a, n / l);
        f = Math.floor(a * B), g = Math.floor(l * B);
      }
      y = document.createElement("canvas");
      const S = y.width = Math.ceil(f * u.sx), x = y.height = Math.ceil(g * u.sy);
      o(this, Ti) || (h = E(this, Gt, Um).call(this, S, x));
      const _ = y.getContext("2d");
      _.filter = this._uiManager.hcmFilter;
      let k = "white", T = "#cfcfd8";
      this._uiManager.hcmFilter !== "none" ? T = "black" : Ak.isDarkMode && (k = "#8f8f9d", T = "#42414d");
      const R = 15, P = R * u.sx, L = R * u.sy, D = new OffscreenCanvas(P * 2, L * 2), N = D.getContext("2d");
      N.fillStyle = k, N.fillRect(0, 0, P * 2, L * 2), N.fillStyle = T, N.fillRect(0, 0, P, L), N.fillRect(P, L, P, L), _.fillStyle = _.createPattern(D, "repeat"), _.fillRect(0, 0, S, x), _.drawImage(h, 0, 0, h.width, h.height, 0, 0, S, x);
    }
    let v = null;
    if (s) {
      let S, x;
      if (u.symmetric && h.width < e && h.height < e)
        S = h.width, x = h.height;
      else if (h = o(this, Ae), a > e || l > e) {
        const T = Math.min(e / a, e / l);
        S = Math.floor(a * T), x = Math.floor(l * T), o(this, Ti) || (h = E(this, Gt, Um).call(this, S, x));
      }
      const k = new OffscreenCanvas(S, x).getContext("2d", {
        willReadFrequently: !0
      });
      k.drawImage(h, 0, 0, h.width, h.height, 0, 0, S, x), v = {
        width: S,
        height: x,
        data: k.getImageData(0, 0, S, x).data
      };
    }
    return {
      canvas: y,
      width: f,
      height: g,
      imageData: v
    };
  }
  static async deserialize(e, n, s) {
    var k;
    let a = null, l = !1;
    if (e instanceof $_) {
      const {
        data: {
          rect: T,
          rotation: R,
          id: P,
          structParent: L,
          popupRef: D,
          richText: N,
          contentsObj: B,
          creationDate: $,
          modificationDate: V
        },
        container: X,
        parent: {
          page: {
            pageNumber: q
          }
        },
        canvas: it
      } = e;
      let lt, rt;
      it ? (delete e.canvas, {
        id: lt,
        bitmap: rt
      } = s.imageManager.getFromCanvas(X.id, it), it.remove()) : (l = !0, e._hasNoCanvas = !0);
      const vt = ((k = await n._structTree.getAriaAttributes(`${hA}${P}`)) == null ? void 0 : k.get("aria-label")) || "";
      a = e = {
        annotationType: Pt.STAMP,
        bitmapId: lt,
        bitmap: rt,
        pageIndex: q - 1,
        rect: T.slice(0),
        rotation: R,
        annotationElementId: P,
        id: P,
        deleted: !1,
        accessibilityData: {
          decorative: !1,
          altText: vt
        },
        isSvg: !1,
        structParent: L,
        popupRef: D,
        richText: N,
        comment: (B == null ? void 0 : B.str) || null,
        creationDate: $,
        modificationDate: V
      };
    }
    const u = await super.deserialize(e, n, s), {
      rect: h,
      bitmap: f,
      bitmapUrl: g,
      bitmapId: y,
      isSvg: v,
      accessibilityData: S
    } = e;
    l ? (s.addMissingCanvas(e.id, u), w(u, lc, !0)) : y && s.imageManager.isValidId(y) ? (w(u, an, y), f && w(u, Ae, f)) : w(u, to, g), w(u, Ti, v);
    const [x, _] = u.pageDimensions;
    return u.width = (h[2] - h[0]) / x, u.height = (h[3] - h[1]) / _, S && (u.altTextData = S), u._initialData = a, e.comment && u.setCommentData(e), w(u, _h, !!a), u;
  }
  serialize(e = !1, n = null) {
    if (this.isEmpty())
      return null;
    if (this.deleted)
      return this.serializeDeleted();
    const s = Object.assign(super.serialize(e), {
      bitmapId: o(this, an),
      isSvg: o(this, Ti)
    });
    if (this.addComment(s), e)
      return s.bitmapUrl = E(this, Gt, Hm).call(this, !0), s.accessibilityData = this.serializeAltText(!0), s.isCopy = !0, s;
    const {
      decorative: a,
      altText: l
    } = this.serializeAltText(!1);
    if (!a && l && (s.accessibilityData = {
      type: "Figure",
      alt: l
    }), this.annotationElementId) {
      const h = E(this, Gt, TC).call(this, s);
      return h.isSame ? null : (h.isSameAltText ? delete s.accessibilityData : s.accessibilityData.structParent = this._initialData.structParent ?? -1, s.id = this.annotationElementId, delete s.bitmapId, s);
    }
    if (n === null)
      return s;
    n.stamps || (n.stamps = /* @__PURE__ */ new Map());
    const u = o(this, Ti) ? (s.rect[2] - s.rect[0]) * (s.rect[3] - s.rect[1]) : null;
    if (!n.stamps.has(o(this, an)))
      n.stamps.set(o(this, an), {
        area: u,
        serialized: s
      }), s.bitmap = E(this, Gt, Hm).call(this, !1);
    else if (o(this, Ti)) {
      const h = n.stamps.get(o(this, an));
      u > h.area && (h.area = u, h.serialized.bitmap.close(), h.serialized.bitmap = E(this, Gt, Hm).call(this, !1));
    }
    return s;
  }
  renderAnnotationElement(e) {
    return this.deleted ? (e.hide(), null) : (e.updateEdited({
      rect: this.getPDFRect(),
      popup: this.comment
    }), null);
  }
}
Ae = new WeakMap(), an = new WeakMap(), Aa = new WeakMap(), to = new WeakMap(), Sa = new WeakMap(), Eh = new WeakMap(), eo = new WeakMap(), lc = new WeakMap(), ar = new WeakMap(), Ti = new WeakMap(), _h = new WeakMap(), Gt = new WeakSet(), jd = function(e, n = !1) {
  if (!e) {
    this.remove();
    return;
  }
  w(this, Ae, e.bitmap), n || (w(this, an, e.id), w(this, Ti, e.isSvg)), e.file && w(this, Eh, e.file.name), E(this, Gt, zm).call(this);
}, $d = function() {
  if (w(this, Aa, null), this._uiManager.enableWaiting(!1), !!o(this, eo)) {
    if (this._uiManager.useNewAltTextWhenAddingImage && this._uiManager.useNewAltTextFlow && o(this, Ae)) {
      this.addEditToolbar().then(() => {
        this._editToolbar.hide(), this._uiManager.editAltText(this, !0);
      });
      return;
    }
    if (!this._uiManager.useNewAltTextWhenAddingImage && this._uiManager.useNewAltTextFlow && o(this, Ae)) {
      this._reportTelemetry({
        action: "pdfjs.image.image_added",
        data: {
          alt_text_modal: !1,
          alt_text_type: "empty"
        }
      });
      try {
        this.mlGuessAltText();
      } catch {
      }
    }
    this.div.focus();
  }
}, Bm = function() {
  if (o(this, an)) {
    this._uiManager.enableWaiting(!0), this._uiManager.imageManager.getFromId(o(this, an)).then((s) => E(this, Gt, jd).call(this, s, !0)).finally(() => E(this, Gt, $d).call(this));
    return;
  }
  if (o(this, to)) {
    const s = o(this, to);
    w(this, to, null), this._uiManager.enableWaiting(!0), w(this, Aa, this._uiManager.imageManager.getFromUrl(s).then((a) => E(this, Gt, jd).call(this, a)).finally(() => E(this, Gt, $d).call(this)));
    return;
  }
  if (o(this, Sa)) {
    const s = o(this, Sa);
    w(this, Sa, null), this._uiManager.enableWaiting(!0), w(this, Aa, this._uiManager.imageManager.getFromFile(s).then((a) => E(this, Gt, jd).call(this, a)).finally(() => E(this, Gt, $d).call(this)));
    return;
  }
  const e = document.createElement("input");
  e.type = "file", e.accept = Vm.join(",");
  const n = this._uiManager._signal;
  w(this, Aa, new Promise((s) => {
    e.addEventListener("change", async () => {
      if (!e.files || e.files.length === 0)
        this.remove();
      else {
        this._uiManager.enableWaiting(!0);
        const a = await this._uiManager.imageManager.getFromFile(e.files[0]);
        this._reportTelemetry({
          action: "pdfjs.image.image_selected",
          data: {
            alt_text_modal: this._uiManager.useNewAltTextFlow
          }
        }), E(this, Gt, jd).call(this, a);
      }
      s();
    }, {
      signal: n
    }), e.addEventListener("cancel", () => {
      this.remove(), s();
    }, {
      signal: n
    });
  }).finally(() => E(this, Gt, $d).call(this))), e.click();
}, zm = function() {
  var f;
  const {
    div: e
  } = this;
  let {
    width: n,
    height: s
  } = o(this, Ae);
  const [a, l] = this.pageDimensions, u = 0.75;
  if (this.width)
    n = this.width * a, s = this.height * l;
  else if (n > u * a || s > u * l) {
    const g = Math.min(u * a / n, u * l / s);
    n *= g, s *= g;
  }
  this._uiManager.enableWaiting(!1);
  const h = w(this, eo, document.createElement("canvas"));
  h.setAttribute("role", "img"), this.addContainer(h), this.width = n / a, this.height = s / l, this.setDims(), (f = this._initialOptions) != null && f.isCentered ? this.center() : this.fixAndSetPosition(), this._initialOptions = null, (!this._uiManager.useNewAltTextWhenAddingImage || !this._uiManager.useNewAltTextFlow || this.annotationElementId) && (e.hidden = !1), E(this, Gt, Jw).call(this), o(this, _h) || (this.parent.addUndoableEditor(this), w(this, _h, !0)), this._reportTelemetry({
    action: "inserted_image"
  }), o(this, Eh) && this.div.setAttribute("aria-description", o(this, Eh)), this.annotationElementId || this._uiManager.a11yAlert("pdfjs-editor-stamp-added-alert");
}, Um = function(e, n) {
  const {
    width: s,
    height: a
  } = o(this, Ae);
  let l = s, u = a, h = o(this, Ae);
  for (; l > 2 * e || u > 2 * n; ) {
    const f = l, g = u;
    l > 2 * e && (l = l >= 16384 ? Math.floor(l / 2) - 1 : Math.ceil(l / 2)), u > 2 * n && (u = u >= 16384 ? Math.floor(u / 2) - 1 : Math.ceil(u / 2));
    const y = new OffscreenCanvas(l, u);
    y.getContext("2d").drawImage(h, 0, 0, f, g, 0, 0, l, u), h = y.transferToImageBitmap();
  }
  return h;
}, Jw = function() {
  const [e, n] = this.parentDimensions, {
    width: s,
    height: a
  } = this, l = new ks(), u = Math.ceil(s * e * l.sx), h = Math.ceil(a * n * l.sy), f = o(this, eo);
  if (!f || f.width === u && f.height === h)
    return;
  f.width = u, f.height = h;
  const g = o(this, Ti) ? o(this, Ae) : E(this, Gt, Um).call(this, u, h), y = f.getContext("2d");
  y.filter = this._uiManager.hcmFilter, y.drawImage(g, 0, 0, g.width, g.height, 0, 0, u, h);
}, Hm = function(e) {
  if (e) {
    if (o(this, Ti)) {
      const a = this._uiManager.imageManager.getSvgUrl(o(this, an));
      if (a)
        return a;
    }
    const n = document.createElement("canvas");
    return {
      width: n.width,
      height: n.height
    } = o(this, Ae), n.getContext("2d").drawImage(o(this, Ae), 0, 0), n.toDataURL();
  }
  if (o(this, Ti)) {
    const [n, s] = this.pageDimensions, a = Math.round(this.width * n * lo.PDF_TO_CSS_UNITS), l = Math.round(this.height * s * lo.PDF_TO_CSS_UNITS), u = new OffscreenCanvas(a, l);
    return u.getContext("2d").drawImage(o(this, Ae), 0, 0, o(this, Ae).width, o(this, Ae).height, 0, 0, a, l), u.transferToImageBitmap();
  }
  return structuredClone(o(this, Ae));
}, TC = function(e) {
  var u;
  const {
    pageIndex: n,
    accessibilityData: {
      altText: s
    }
  } = this._initialData, a = e.pageIndex === n, l = (((u = e.accessibilityData) == null ? void 0 : u.alt) || "") === s;
  return {
    isSame: !this.hasEditedComment && !this._hasBeenMoved && !this._hasBeenResized && a && l,
    isSameAltText: l
  };
}, Z(Zw, "_type", "stamp"), Z(Zw, "_editorType", Pt.STAMP);
var cc, Ch, ba, Ea, no, si, _a, xh, Th, Cs, io, ln, so, Ca, kh, wt, xa, le, tA, kC, Bs, eA, nA, jm;
const ps = class ps {
  constructor({
    uiManager: t,
    pageIndex: e,
    div: n,
    structTreeLayer: s,
    accessibilityManager: a,
    annotationLayer: l,
    drawLayer: u,
    textLayer: h,
    viewport: f,
    l10n: g
  }) {
    b(this, le);
    b(this, cc);
    b(this, Ch, !1);
    b(this, ba, null);
    b(this, Ea, null);
    b(this, no, null);
    b(this, si, /* @__PURE__ */ new Map());
    b(this, _a, !1);
    b(this, xh, !1);
    b(this, Th, !1);
    b(this, Cs, null);
    b(this, io, null);
    b(this, ln, null);
    b(this, so, null);
    b(this, Ca, null);
    b(this, kh, -1);
    b(this, wt);
    const y = [...o(ps, xa).values()];
    if (!ps._initialized) {
      ps._initialized = !0;
      for (const v of y)
        v.initialize(g, t);
    }
    t.registerEditorTypes(y), w(this, wt, t), this.pageIndex = e, this.div = n, w(this, cc, a), w(this, ba, l), this.viewport = f, w(this, ln, h), this.drawLayer = u, this._structTree = s, o(this, wt).addLayer(this);
  }
  get isEmpty() {
    return o(this, si).size === 0;
  }
  get isInvisible() {
    return this.isEmpty && o(this, wt).getMode() === Pt.NONE;
  }
  updateToolbar(t) {
    o(this, wt).updateToolbar(t);
  }
  updateMode(t = o(this, wt).getMode()) {
    switch (E(this, le, jm).call(this), t) {
      case Pt.NONE:
        this.div.classList.toggle("nonEditing", !0), this.disableTextSelection(), this.togglePointerEvents(!1), this.toggleAnnotationLayerPointerEvents(!0), this.disableClick();
        return;
      case Pt.INK:
        this.disableTextSelection(), this.togglePointerEvents(!0), this.enableClick();
        break;
      case Pt.HIGHLIGHT:
        this.enableTextSelection(), this.togglePointerEvents(!1), this.disableClick();
        break;
      default:
        this.disableTextSelection(), this.togglePointerEvents(!0), this.enableClick();
    }
    this.toggleAnnotationLayerPointerEvents(!1);
    const {
      classList: e
    } = this.div;
    if (e.toggle("nonEditing", !1), t === Pt.POPUP)
      e.toggle("commentEditing", !0);
    else {
      e.toggle("commentEditing", !1);
      for (const n of o(ps, xa).values())
        e.toggle(`${n._type}Editing`, t === n._editorType);
    }
    this.div.hidden = !1;
  }
  hasTextLayer(t) {
    var e;
    return t === ((e = o(this, ln)) == null ? void 0 : e.div);
  }
  setEditingState(t) {
    o(this, wt).setEditingState(t);
  }
  addCommands(t) {
    o(this, wt).addCommands(t);
  }
  cleanUndoStack(t) {
    o(this, wt).cleanUndoStack(t);
  }
  toggleDrawing(t = !1) {
    this.div.classList.toggle("drawing", !t);
  }
  togglePointerEvents(t = !1) {
    this.div.classList.toggle("disabled", !t);
  }
  toggleAnnotationLayerPointerEvents(t = !1) {
    var e;
    (e = o(this, ba)) == null || e.div.classList.toggle("disabled", !t);
  }
  async enable() {
    var n;
    w(this, Th, !0), this.div.tabIndex = 0, this.togglePointerEvents(!0), this.div.classList.toggle("nonEditing", !1), (n = o(this, Ca)) == null || n.abort(), w(this, Ca, null);
    const t = /* @__PURE__ */ new Set();
    for (const s of o(this, le, tA))
      s.enableEditing(), s.show(!0), s.annotationElementId && (o(this, wt).removeChangedExistingAnnotation(s), t.add(s.annotationElementId));
    const e = o(this, ba);
    if (e)
      for (const s of e.getEditableAnnotations()) {
        if (s.hide(), o(this, wt).isDeletedAnnotationElement(s.data.id) || t.has(s.data.id))
          continue;
        const a = await this.deserialize(s);
        a && (this.addOrRebuild(a), a.enableEditing());
      }
    w(this, Th, !1), o(this, wt)._eventBus.dispatch("editorsrendered", {
      source: this,
      pageNumber: this.pageIndex + 1
    });
  }
  disable() {
    var n;
    if (w(this, xh, !0), this.div.tabIndex = -1, this.togglePointerEvents(!1), this.div.classList.toggle("nonEditing", !0), o(this, ln) && !o(this, Ca)) {
      w(this, Ca, new AbortController());
      const s = o(this, wt).combinedSignal(o(this, Ca));
      o(this, ln).div.addEventListener("pointerdown", (a) => {
        const {
          clientX: u,
          clientY: h,
          timeStamp: f
        } = a, g = o(this, kh);
        if (f - g > 500) {
          w(this, kh, f);
          return;
        }
        w(this, kh, -1);
        const {
          classList: y
        } = this.div;
        y.toggle("getElements", !0);
        const v = document.elementsFromPoint(u, h);
        if (y.toggle("getElements", !1), !this.div.contains(v[0]))
          return;
        let S;
        const x = new RegExp(`^${Wd}[0-9]+$`);
        for (const k of v)
          if (x.test(k.id)) {
            S = k.id;
            break;
          }
        if (!S)
          return;
        const _ = o(this, si).get(S);
        (_ == null ? void 0 : _.annotationElementId) === null && (a.stopPropagation(), a.preventDefault(), _.dblclick(a));
      }, {
        signal: s,
        capture: !0
      });
    }
    const t = o(this, ba);
    if (t) {
      const s = /* @__PURE__ */ new Map(), a = /* @__PURE__ */ new Map();
      for (const u of o(this, le, tA)) {
        if (u.disableEditing(), !u.annotationElementId) {
          u.updateFakeAnnotationElement(t);
          continue;
        }
        if (u.serialize() !== null) {
          s.set(u.annotationElementId, u);
          continue;
        } else
          a.set(u.annotationElementId, u);
        (n = this.getEditableAnnotation(u.annotationElementId)) == null || n.show(), u.remove();
      }
      const l = t.getEditableAnnotations();
      for (const u of l) {
        const {
          id: h
        } = u.data;
        if (o(this, wt).isDeletedAnnotationElement(h)) {
          u.updateEdited({
            deleted: !0
          });
          continue;
        }
        let f = a.get(h);
        if (f) {
          f.resetAnnotationElement(u), f.show(!1), u.show();
          continue;
        }
        f = s.get(h), f && (o(this, wt).addChangedExistingAnnotation(f), f.renderAnnotationElement(u) && f.show(!1)), u.show();
      }
    }
    E(this, le, jm).call(this), this.isEmpty && (this.div.hidden = !0);
    const {
      classList: e
    } = this.div;
    for (const s of o(ps, xa).values())
      e.remove(`${s._type}Editing`);
    this.disableTextSelection(), this.toggleAnnotationLayerPointerEvents(!0), w(this, xh, !1);
  }
  getEditableAnnotation(t) {
    var e;
    return ((e = o(this, ba)) == null ? void 0 : e.getEditableAnnotation(t)) || null;
  }
  setActiveEditor(t) {
    o(this, wt).getActive() !== t && o(this, wt).setActiveEditor(t);
  }
  enableTextSelection() {
    var t;
    if (this.div.tabIndex = -1, (t = o(this, ln)) != null && t.div && !o(this, so)) {
      w(this, so, new AbortController());
      const e = o(this, wt).combinedSignal(o(this, so));
      o(this, ln).div.addEventListener("pointerdown", E(this, le, kC).bind(this), {
        signal: e
      }), o(this, ln).div.classList.add("highlighting");
    }
  }
  disableTextSelection() {
    var t;
    this.div.tabIndex = 0, (t = o(this, ln)) != null && t.div && o(this, so) && (o(this, so).abort(), w(this, so, null), o(this, ln).div.classList.remove("highlighting"));
  }
  enableClick() {
    if (o(this, Ea))
      return;
    w(this, Ea, new AbortController());
    const t = o(this, wt).combinedSignal(o(this, Ea));
    this.div.addEventListener("pointerdown", this.pointerdown.bind(this), {
      signal: t
    });
    const e = this.pointerup.bind(this);
    this.div.addEventListener("pointerup", e, {
      signal: t
    }), this.div.addEventListener("pointercancel", e, {
      signal: t
    });
  }
  disableClick() {
    var t;
    (t = o(this, Ea)) == null || t.abort(), w(this, Ea, null);
  }
  attach(t) {
    o(this, si).set(t.id, t);
    const {
      annotationElementId: e
    } = t;
    e && o(this, wt).isDeletedAnnotationElement(e) && o(this, wt).removeDeletedAnnotationElement(t);
  }
  detach(t) {
    var e;
    o(this, si).delete(t.id), (e = o(this, cc)) == null || e.removePointerInTextLayer(t.contentDiv), !o(this, xh) && t.annotationElementId && o(this, wt).addDeletedAnnotationElement(t);
  }
  remove(t) {
    this.detach(t), o(this, wt).removeEditor(t), t.div.remove(), t.isAttachedToDOM = !1;
  }
  changeParent(t) {
    var e;
    t.parent !== this && (t.parent && t.annotationElementId && (o(this, wt).addDeletedAnnotationElement(t.annotationElementId), fe.deleteAnnotationElement(t), t.annotationElementId = null), this.attach(t), (e = t.parent) == null || e.detach(t), t.setParent(this), t.div && t.isAttachedToDOM && (t.div.remove(), this.div.append(t.div)));
  }
  add(t) {
    if (!(t.parent === this && t.isAttachedToDOM)) {
      if (this.changeParent(t), o(this, wt).addEditor(t), this.attach(t), !t.isAttachedToDOM) {
        const e = t.render();
        this.div.append(e), t.isAttachedToDOM = !0;
      }
      t.fixAndSetPosition(), t.onceAdded(!o(this, Th)), o(this, wt).addToAnnotationStorage(t), t._reportTelemetry(t.telemetryInitialData);
    }
  }
  moveEditorInDOM(t) {
    var n;
    if (!t.isAttachedToDOM)
      return;
    const {
      activeElement: e
    } = document;
    t.div.contains(e) && !o(this, no) && (t._focusEventsAllowed = !1, w(this, no, setTimeout(() => {
      w(this, no, null), t.div.contains(document.activeElement) ? t._focusEventsAllowed = !0 : (t.div.addEventListener("focusin", () => {
        t._focusEventsAllowed = !0;
      }, {
        once: !0,
        signal: o(this, wt)._signal
      }), e.focus());
    }, 0))), t._structTreeParentId = (n = o(this, cc)) == null ? void 0 : n.moveElementInDOM(this.div, t.div, t.contentDiv, !0);
  }
  addOrRebuild(t) {
    t.needsToBeRebuilt() ? (t.parent || (t.parent = this), t.rebuild(), t.show()) : this.add(t);
  }
  addUndoableEditor(t) {
    const e = () => t._uiManager.rebuild(t), n = () => {
      t.remove();
    };
    this.addCommands({
      cmd: e,
      undo: n,
      mustExec: !1
    });
  }
  getEditorByUID(t) {
    for (const e of o(this, si).values())
      if (e.uid === t)
        return e;
    return null;
  }
  getNextId() {
    return o(this, wt).getId();
  }
  combinedSignal(t) {
    return o(this, wt).combinedSignal(t);
  }
  canCreateNewEmptyEditor() {
    var t;
    return (t = o(this, le, Bs)) == null ? void 0 : t.canCreateNewEmptyEditor();
  }
  async pasteEditor(t, e) {
    this.updateToolbar(t), await o(this, wt).updateMode(t.mode);
    const {
      offsetX: n,
      offsetY: s
    } = E(this, le, nA).call(this), a = this.getNextId(), l = E(this, le, eA).call(this, {
      parent: this,
      id: a,
      x: n,
      y: s,
      uiManager: o(this, wt),
      isCentered: !0,
      ...e
    });
    l && this.add(l);
  }
  async deserialize(t) {
    var e;
    return await ((e = o(ps, xa).get(t.annotationType ?? t.annotationEditorType)) == null ? void 0 : e.deserialize(t, this, o(this, wt))) || null;
  }
  createAndAddNewEditor(t, e, n = {}) {
    const s = this.getNextId(), a = E(this, le, eA).call(this, {
      parent: this,
      id: s,
      x: t.offsetX,
      y: t.offsetY,
      uiManager: o(this, wt),
      isCentered: e,
      ...n
    });
    return a && this.add(a), a;
  }
  get boundingClientRect() {
    return this.div.getBoundingClientRect();
  }
  addNewEditor(t = {}) {
    this.createAndAddNewEditor(E(this, le, nA).call(this), !0, t);
  }
  setSelected(t) {
    o(this, wt).setSelected(t);
  }
  toggleSelected(t) {
    o(this, wt).toggleSelected(t);
  }
  unselect(t) {
    o(this, wt).unselect(t);
  }
  pointerup(t) {
    var s;
    const {
      isMac: e
    } = cn.platform;
    if (t.button !== 0 || t.ctrlKey && e || t.target !== this.div || !o(this, _a) || (w(this, _a, !1), (s = o(this, le, Bs)) != null && s.isDrawer && o(this, le, Bs).supportMultipleDrawings))
      return;
    if (!o(this, Ch)) {
      w(this, Ch, !0);
      return;
    }
    const n = o(this, wt).getMode();
    if (n === Pt.STAMP || n === Pt.SIGNATURE) {
      o(this, wt).unselectAll();
      return;
    }
    this.createAndAddNewEditor(t, !1);
  }
  pointerdown(t) {
    var s;
    if (o(this, wt).getMode() === Pt.HIGHLIGHT && this.enableTextSelection(), o(this, _a)) {
      w(this, _a, !1);
      return;
    }
    const {
      isMac: e
    } = cn.platform;
    if (t.button !== 0 || t.ctrlKey && e || t.target !== this.div)
      return;
    if (w(this, _a, !0), (s = o(this, le, Bs)) != null && s.isDrawer) {
      this.startDrawingSession(t);
      return;
    }
    const n = o(this, wt).getActive();
    w(this, Ch, !n || n.isEmpty());
  }
  startDrawingSession(t) {
    if (this.div.focus({
      preventScroll: !0
    }), o(this, Cs)) {
      o(this, le, Bs).startDrawing(this, o(this, wt), !1, t);
      return;
    }
    o(this, wt).setCurrentDrawingSession(this), w(this, Cs, new AbortController());
    const e = o(this, wt).combinedSignal(o(this, Cs));
    this.div.addEventListener("blur", ({
      relatedTarget: n
    }) => {
      n && !this.div.contains(n) && (w(this, io, null), this.commitOrRemove());
    }, {
      signal: e
    }), o(this, le, Bs).startDrawing(this, o(this, wt), !1, t);
  }
  pause(t) {
    if (t) {
      const {
        activeElement: e
      } = document;
      this.div.contains(e) && w(this, io, e);
      return;
    }
    o(this, io) && setTimeout(() => {
      var e;
      (e = o(this, io)) == null || e.focus(), w(this, io, null);
    }, 0);
  }
  endDrawingSession(t = !1) {
    return o(this, Cs) ? (o(this, wt).setCurrentDrawingSession(null), o(this, Cs).abort(), w(this, Cs, null), w(this, io, null), o(this, le, Bs).endDrawing(t)) : null;
  }
  findNewParent(t, e, n) {
    const s = o(this, wt).findParent(e, n);
    return s === null || s === this ? !1 : (s.changeParent(t), !0);
  }
  commitOrRemove() {
    return o(this, Cs) ? (this.endDrawingSession(), !0) : !1;
  }
  onScaleChanging() {
    o(this, Cs) && o(this, le, Bs).onScaleChangingWhenDrawing(this);
  }
  destroy() {
    var t, e;
    this.commitOrRemove(), ((t = o(this, wt).getActive()) == null ? void 0 : t.parent) === this && (o(this, wt).commitOrRemove(), o(this, wt).setActiveEditor(null)), o(this, no) && (clearTimeout(o(this, no)), w(this, no, null));
    for (const n of o(this, si).values())
      (e = o(this, cc)) == null || e.removePointerInTextLayer(n.contentDiv), n.setParent(null), n.isAttachedToDOM = !1, n.div.remove();
    this.div = null, o(this, si).clear(), o(this, wt).removeLayer(this);
  }
  render({
    viewport: t
  }) {
    this.viewport = t, Ta(this.div, t);
    for (const e of o(this, wt).getEditors(this.pageIndex))
      this.add(e), e.rebuild();
    this.updateMode();
  }
  update({
    viewport: t
  }) {
    o(this, wt).commitOrRemove(), E(this, le, jm).call(this);
    const e = this.viewport.rotation, n = t.rotation;
    if (this.viewport = t, Ta(this.div, {
      rotation: n
    }), e !== n)
      for (const s of o(this, si).values())
        s.rotate(n);
  }
  get pageDimensions() {
    const {
      pageWidth: t,
      pageHeight: e
    } = this.viewport.rawDims;
    return [t, e];
  }
  get scale() {
    return o(this, wt).viewParameters.realScale;
  }
};
cc = new WeakMap(), Ch = new WeakMap(), ba = new WeakMap(), Ea = new WeakMap(), no = new WeakMap(), si = new WeakMap(), _a = new WeakMap(), xh = new WeakMap(), Th = new WeakMap(), Cs = new WeakMap(), io = new WeakMap(), ln = new WeakMap(), so = new WeakMap(), Ca = new WeakMap(), kh = new WeakMap(), wt = new WeakMap(), xa = new WeakMap(), le = new WeakSet(), tA = function() {
  return o(this, si).size !== 0 ? o(this, si).values() : o(this, wt).getEditors(this.pageIndex);
}, kC = function(t) {
  o(this, wt).unselectAll();
  const {
    target: e
  } = t;
  if (e === o(this, ln).div || (e.getAttribute("role") === "img" || e.classList.contains("endOfContent")) && o(this, ln).div.contains(e)) {
    const {
      isMac: n
    } = cn.platform;
    if (t.button !== 0 || t.ctrlKey && n)
      return;
    o(this, wt).showAllEditors("highlight", !0, !0), o(this, ln).div.classList.add("free"), this.toggleDrawing(), Qm.startHighlighting(this, o(this, wt).direction === "ltr", {
      target: o(this, ln).div,
      x: t.x,
      y: t.y
    }), o(this, ln).div.addEventListener("pointerup", () => {
      o(this, ln).div.classList.remove("free"), this.toggleDrawing(!0);
    }, {
      once: !0,
      signal: o(this, wt)._signal
    }), t.preventDefault();
  }
}, Bs = function() {
  return o(ps, xa).get(o(this, wt).getMode());
}, eA = function(t) {
  const e = o(this, le, Bs);
  return e ? new e.prototype.constructor(t) : null;
}, nA = function() {
  const {
    x: t,
    y: e,
    width: n,
    height: s
  } = this.boundingClientRect, a = Math.max(0, t), l = Math.max(0, e), u = Math.min(window.innerWidth, t + n), h = Math.min(window.innerHeight, e + s), f = (a + u) / 2 - t, g = (l + h) / 2 - e, [y, v] = this.viewport.rotation % 180 === 0 ? [f, g] : [g, f];
  return {
    offsetX: y,
    offsetY: v
  };
}, jm = function() {
  for (const t of o(this, si).values())
    t.isEmpty() && t.remove();
}, Z(ps, "_initialized", !1), b(ps, xa, new Map([Fw, Ww, Zw, Qm, Qw].map((t) => [t._editorType, t])));
let Jm = ps;
var xs, Un, uc, Dp, wy, PC, ur, iA, RC, sA;
const Xe = class Xe {
  constructor({
    pageIndex: t
  }) {
    b(this, ur);
    b(this, xs, null);
    b(this, Un, /* @__PURE__ */ new Map());
    b(this, uc, /* @__PURE__ */ new Map());
    this.pageIndex = t;
  }
  setParent(t) {
    if (!o(this, xs)) {
      w(this, xs, t);
      return;
    }
    if (o(this, xs) !== t) {
      if (o(this, Un).size > 0)
        for (const e of o(this, Un).values())
          e.remove(), t.append(e);
      w(this, xs, t);
    }
  }
  static get _svgFactory() {
    return It(this, "_svgFactory", new Zd());
  }
  draw(t, e = !1, n = !1) {
    const s = Ge(Xe, Dp)._++, a = E(this, ur, iA).call(this), l = Xe._svgFactory.createElement("defs");
    a.append(l);
    const u = Xe._svgFactory.createElement("path");
    l.append(u);
    const h = `path_p${this.pageIndex}_${s}`;
    u.setAttribute("id", h), u.setAttribute("vector-effect", "non-scaling-stroke"), e && o(this, uc).set(s, u);
    const f = n ? E(this, ur, RC).call(this, l, h) : null, g = Xe._svgFactory.createElement("use");
    return a.append(g), g.setAttribute("href", `#${h}`), this.updateProperties(a, t), o(this, Un).set(s, a), {
      id: s,
      clipPathId: `url(#${f})`
    };
  }
  drawOutline(t, e) {
    const n = Ge(Xe, Dp)._++, s = E(this, ur, iA).call(this), a = Xe._svgFactory.createElement("defs");
    s.append(a);
    const l = Xe._svgFactory.createElement("path");
    a.append(l);
    const u = `path_p${this.pageIndex}_${n}`;
    l.setAttribute("id", u), l.setAttribute("vector-effect", "non-scaling-stroke");
    let h;
    if (e) {
      const y = Xe._svgFactory.createElement("mask");
      a.append(y), h = `mask_p${this.pageIndex}_${n}`, y.setAttribute("id", h), y.setAttribute("maskUnits", "objectBoundingBox");
      const v = Xe._svgFactory.createElement("rect");
      y.append(v), v.setAttribute("width", "1"), v.setAttribute("height", "1"), v.setAttribute("fill", "white");
      const S = Xe._svgFactory.createElement("use");
      y.append(S), S.setAttribute("href", `#${u}`), S.setAttribute("stroke", "none"), S.setAttribute("fill", "black"), S.setAttribute("fill-rule", "nonzero"), S.classList.add("mask");
    }
    const f = Xe._svgFactory.createElement("use");
    s.append(f), f.setAttribute("href", `#${u}`), h && f.setAttribute("mask", `url(#${h})`);
    const g = f.cloneNode();
    return s.append(g), f.classList.add("mainOutline"), g.classList.add("secondaryOutline"), this.updateProperties(s, t), o(this, Un).set(n, s), n;
  }
  finalizeDraw(t, e) {
    o(this, uc).delete(t), this.updateProperties(t, e);
  }
  updateProperties(t, e) {
    var h;
    if (!e)
      return;
    const {
      root: n,
      bbox: s,
      rootClass: a,
      path: l
    } = e, u = typeof t == "number" ? o(this, Un).get(t) : t;
    if (u) {
      if (n && E(this, ur, sA).call(this, u, n), s && E(h = Xe, wy, PC).call(h, u, s), a) {
        const {
          classList: f
        } = u;
        for (const [g, y] of Object.entries(a))
          f.toggle(g, y);
      }
      if (l) {
        const g = u.firstChild.firstChild;
        E(this, ur, sA).call(this, g, l);
      }
    }
  }
  updateParent(t, e) {
    if (e === this)
      return;
    const n = o(this, Un).get(t);
    n && (o(e, xs).append(n), o(this, Un).delete(t), o(e, Un).set(t, n));
  }
  remove(t) {
    o(this, uc).delete(t), o(this, xs) !== null && (o(this, Un).get(t).remove(), o(this, Un).delete(t));
  }
  destroy() {
    w(this, xs, null);
    for (const t of o(this, Un).values())
      t.remove();
    o(this, Un).clear(), o(this, uc).clear();
  }
};
xs = new WeakMap(), Un = new WeakMap(), uc = new WeakMap(), Dp = new WeakMap(), wy = new WeakSet(), PC = function(t, [e, n, s, a]) {
  const {
    style: l
  } = t;
  l.top = `${100 * n}%`, l.left = `${100 * e}%`, l.width = `${100 * s}%`, l.height = `${100 * a}%`;
}, ur = new WeakSet(), iA = function() {
  const t = Xe._svgFactory.create(1, 1, !0);
  return o(this, xs).append(t), t.setAttribute("aria-hidden", !0), t;
}, RC = function(t, e) {
  const n = Xe._svgFactory.createElement("clipPath");
  t.append(n);
  const s = `clip_${e}`;
  n.setAttribute("id", s), n.setAttribute("clipPathUnits", "objectBoundingBox");
  const a = Xe._svgFactory.createElement("use");
  return n.append(a), a.setAttribute("href", `#${e}`), a.classList.add("clip"), s;
}, sA = function(t, e) {
  for (const [n, s] of Object.entries(e))
    s === null ? t.removeAttribute(n) : t.setAttribute(n, s);
}, b(Xe, wy), b(Xe, Dp, 0);
let ty = Xe;
globalThis._pdfjsTestingUtils = {
  HighlightOutliner: Ow
};
globalThis.pdfjsLib = {
  AbortException: ao,
  AnnotationEditorLayer: Jm,
  AnnotationEditorParamsType: jt,
  AnnotationEditorType: Pt,
  AnnotationEditorUIManager: ka,
  AnnotationLayer: SA,
  AnnotationMode: lr,
  AnnotationType: Oe,
  applyOpacity: T1,
  build: M_,
  ColorPicker: Qd,
  createValidAbsoluteUrl: lA,
  CSSConstants: x1,
  DOMSVGFactory: Zd,
  DrawLayer: ty,
  FeatureTest: cn,
  fetchData: Lh,
  findContrastColor: k1,
  getDocument: yA,
  getFilenameFromUrl: E1,
  getPdfFilenameFromUrl: _1,
  getRGB: Dh,
  getUuid: uA,
  getXfaPageViewport: C1,
  GlobalWorkerOptions: ts,
  ImageKind: Vd,
  InvalidPDFException: $m,
  isDataScheme: Op,
  isPdfFile: Ey,
  isValidExplicitDest: a_,
  MathClamp: Ln,
  noContextMenu: Li,
  normalizeUnicode: S1,
  OPS: Ph,
  OutputScale: ks,
  PasswordResponses: w1,
  PDFDataRangeTransport: vA,
  PDFDateString: qd,
  PDFWorker: Rh,
  PermissionFlag: v1,
  PixelsPerInch: lo,
  RenderingCancelledException: by,
  renderRichText: fA,
  ResponseException: Gd,
  setLayerDimensions: Ta,
  shadow: It,
  SignatureExtractor: cr,
  stopEvent: Ce,
  SupportedImageMimeTypes: Vm,
  TextLayer: Kd,
  TouchManager: Yd,
  updateUrlHash: cA,
  Util: ft,
  VerbosityLevel: Ip,
  version: R_,
  XfaLayer: dA
};
const XP = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  AbortException: ao,
  AnnotationEditorLayer: Jm,
  AnnotationEditorParamsType: jt,
  AnnotationEditorType: Pt,
  AnnotationEditorUIManager: ka,
  AnnotationLayer: SA,
  AnnotationMode: lr,
  AnnotationType: Oe,
  CSSConstants: x1,
  ColorPicker: Qd,
  DOMSVGFactory: Zd,
  DrawLayer: ty,
  FeatureTest: cn,
  GlobalWorkerOptions: ts,
  ImageKind: Vd,
  InvalidPDFException: $m,
  MathClamp: Ln,
  OPS: Ph,
  OutputScale: ks,
  PDFDataRangeTransport: vA,
  PDFDateString: qd,
  PDFWorker: Rh,
  PasswordResponses: w1,
  PermissionFlag: v1,
  PixelsPerInch: lo,
  RenderingCancelledException: by,
  ResponseException: Gd,
  SignatureExtractor: cr,
  SupportedImageMimeTypes: Vm,
  TextLayer: Kd,
  TouchManager: Yd,
  Util: ft,
  VerbosityLevel: Ip,
  XfaLayer: dA,
  applyOpacity: T1,
  build: M_,
  createValidAbsoluteUrl: lA,
  fetchData: Lh,
  findContrastColor: k1,
  getDocument: yA,
  getFilenameFromUrl: E1,
  getPdfFilenameFromUrl: _1,
  getRGB: Dh,
  getUuid: uA,
  getXfaPageViewport: C1,
  isDataScheme: Op,
  isPdfFile: Ey,
  isValidExplicitDest: a_,
  noContextMenu: Li,
  normalizeUnicode: S1,
  renderRichText: fA,
  setLayerDimensions: Ta,
  shadow: It,
  stopEvent: Ce,
  updateUrlHash: cA,
  version: R_
}, Symbol.toStringTag, { value: "Module" }));
function MC(p) {
  var t, e, n = "";
  if (typeof p == "string" || typeof p == "number") n += p;
  else if (typeof p == "object") if (Array.isArray(p)) {
    var s = p.length;
    for (t = 0; t < s; t++) p[t] && (e = MC(p[t])) && (n && (n += " "), n += e);
  } else for (e in p) p[e] && (n && (n += " "), n += e);
  return n;
}
function xy() {
  for (var p, t, e = 0, n = "", s = arguments.length; e < s; e++) (p = arguments[e]) && (t = MC(p)) && (n && (n += " "), n += t);
  return n;
}
var n1 = Object.prototype.hasOwnProperty;
function i1(p, t, e) {
  for (e of p.keys())
    if (eu(e, t)) return e;
}
function eu(p, t) {
  var e, n, s;
  if (p === t) return !0;
  if (p && t && (e = p.constructor) === t.constructor) {
    if (e === Date) return p.getTime() === t.getTime();
    if (e === RegExp) return p.toString() === t.toString();
    if (e === Array) {
      if ((n = p.length) === t.length)
        for (; n-- && eu(p[n], t[n]); ) ;
      return n === -1;
    }
    if (e === Set) {
      if (p.size !== t.size)
        return !1;
      for (n of p)
        if (s = n, s && typeof s == "object" && (s = i1(t, s), !s) || !t.has(s)) return !1;
      return !0;
    }
    if (e === Map) {
      if (p.size !== t.size)
        return !1;
      for (n of p)
        if (s = n[0], s && typeof s == "object" && (s = i1(t, s), !s) || !eu(n[1], t.get(s)))
          return !1;
      return !0;
    }
    if (e === ArrayBuffer)
      p = new Uint8Array(p), t = new Uint8Array(t);
    else if (e === DataView) {
      if ((n = p.byteLength) === t.byteLength)
        for (; n-- && p.getInt8(n) === t.getInt8(n); ) ;
      return n === -1;
    }
    if (ArrayBuffer.isView(p)) {
      if ((n = p.byteLength) === t.byteLength)
        for (; n-- && p[n] === t[n]; ) ;
      return n === -1;
    }
    if (!e || typeof p == "object") {
      n = 0;
      for (e in p)
        if (n1.call(p, e) && ++n && !n1.call(t, e) || !(e in t) || !eu(p[e], t[e])) return !1;
      return Object.keys(t).length === n;
    }
  }
  return p !== p && t !== t;
}
function Hp(p) {
  let t = !1;
  return {
    promise: new Promise((n, s) => {
      p.then((a) => !t && n(a)).catch((a) => !t && s(a));
    }),
    cancel() {
      t = !0;
    }
  };
}
const qP = ["onCopy", "onCut", "onPaste"], YP = [
  "onCompositionEnd",
  "onCompositionStart",
  "onCompositionUpdate"
], KP = ["onFocus", "onBlur"], QP = ["onInput", "onInvalid", "onReset", "onSubmit"], ZP = ["onLoad", "onError"], JP = ["onKeyDown", "onKeyPress", "onKeyUp"], tR = [
  "onAbort",
  "onCanPlay",
  "onCanPlayThrough",
  "onDurationChange",
  "onEmptied",
  "onEncrypted",
  "onEnded",
  "onError",
  "onLoadedData",
  "onLoadedMetadata",
  "onLoadStart",
  "onPause",
  "onPlay",
  "onPlaying",
  "onProgress",
  "onRateChange",
  "onSeeked",
  "onSeeking",
  "onStalled",
  "onSuspend",
  "onTimeUpdate",
  "onVolumeChange",
  "onWaiting"
], eR = [
  "onClick",
  "onContextMenu",
  "onDoubleClick",
  "onMouseDown",
  "onMouseEnter",
  "onMouseLeave",
  "onMouseMove",
  "onMouseOut",
  "onMouseOver",
  "onMouseUp"
], nR = [
  "onDrag",
  "onDragEnd",
  "onDragEnter",
  "onDragExit",
  "onDragLeave",
  "onDragOver",
  "onDragStart",
  "onDrop"
], iR = ["onSelect"], sR = ["onTouchCancel", "onTouchEnd", "onTouchMove", "onTouchStart"], rR = [
  "onPointerDown",
  "onPointerMove",
  "onPointerUp",
  "onPointerCancel",
  "onGotPointerCapture",
  "onLostPointerCapture",
  "onPointerEnter",
  "onPointerLeave",
  "onPointerOver",
  "onPointerOut"
], oR = ["onScroll"], aR = ["onWheel"], lR = [
  "onAnimationStart",
  "onAnimationEnd",
  "onAnimationIteration"
], cR = ["onTransitionEnd"], uR = ["onToggle"], hR = ["onChange"], dR = [
  ...qP,
  ...YP,
  ...KP,
  ...QP,
  ...ZP,
  ...JP,
  ...tR,
  ...eR,
  ...nR,
  ...iR,
  ...sR,
  ...rR,
  ...oR,
  ...aR,
  ...lR,
  ...cR,
  ...hR,
  ...uR
];
function LC(p, t) {
  const e = {};
  for (const n of dR) {
    const s = p[n];
    s && (t ? e[n] = (a) => s(a, t(n)) : e[n] = s);
  }
  return e;
}
var fR = "Invariant failed";
function ae(p, t) {
  if (!p)
    throw new Error(fR);
}
var L0, s1;
function pR() {
  if (s1) return L0;
  s1 = 1;
  var p = function() {
  };
  return L0 = p, L0;
}
var gR = pR();
const Dn = /* @__PURE__ */ HT(gR), DC = nt.createContext(null), mR = "noopener noreferrer nofollow";
class yR {
  constructor() {
    this.externalLinkEnabled = !0, this.externalLinkRel = void 0, this.externalLinkTarget = void 0, this.isInPresentationMode = !1, this.pdfDocument = void 0, this.pdfViewer = void 0;
  }
  setDocument(t) {
    this.pdfDocument = t;
  }
  setViewer(t) {
    this.pdfViewer = t;
  }
  setExternalLinkRel(t) {
    this.externalLinkRel = t;
  }
  setExternalLinkTarget(t) {
    this.externalLinkTarget = t;
  }
  setHash() {
  }
  setHistory() {
  }
  get pagesCount() {
    return this.pdfDocument ? this.pdfDocument.numPages : 0;
  }
  get page() {
    return ae(this.pdfViewer), this.pdfViewer.currentPageNumber || 0;
  }
  set page(t) {
    ae(this.pdfViewer), this.pdfViewer.currentPageNumber = t;
  }
  get rotation() {
    return 0;
  }
  set rotation(t) {
  }
  addLinkAttributes(t, e, n) {
    t.href = e, t.rel = this.externalLinkRel || mR, t.target = n ? "_blank" : this.externalLinkTarget || "";
  }
  goToDestination(t) {
    return new Promise((e) => {
      ae(this.pdfDocument), ae(t), typeof t == "string" ? this.pdfDocument.getDestination(t).then(e) : Array.isArray(t) ? e(t) : t.then(e);
    }).then((e) => {
      ae(Array.isArray(e));
      const n = e[0];
      new Promise((s) => {
        ae(this.pdfDocument), n instanceof Object ? this.pdfDocument.getPageIndex(n).then((a) => {
          s(a);
        }).catch(() => {
          ae(!1);
        }) : typeof n == "number" ? s(n) : ae(!1);
      }).then((s) => {
        const a = s + 1;
        ae(this.pdfViewer), ae(a >= 1 && a <= this.pagesCount), this.pdfViewer.scrollPageIntoView({
          dest: e,
          pageIndex: s,
          pageNumber: a
        });
      });
    });
  }
  goToPage(t) {
    const e = t - 1;
    ae(this.pdfViewer), ae(t >= 1 && t <= this.pagesCount), this.pdfViewer.scrollPageIntoView({
      pageIndex: e,
      pageNumber: t
    });
  }
  goToXY() {
  }
  cachePageRef() {
  }
  getDestinationHash() {
    return "#";
  }
  getAnchorUrl() {
    return "#";
  }
  executeNamedAction() {
  }
  executeSetOCGState() {
  }
  isPageVisible() {
    return !0;
  }
  isPageCached() {
    return !0;
  }
  navigateTo(t) {
    this.goToDestination(t);
  }
}
function nu({ children: p, type: t }) {
  return St.jsx("div", { className: `react-pdf__message react-pdf__message--${t}`, children: p });
}
const r1 = {
  NEED_PASSWORD: 1,
  INCORRECT_PASSWORD: 2
};
function vR(p, t) {
  switch (t.type) {
    case "RESOLVE":
      return { value: t.value, error: void 0 };
    case "REJECT":
      return { value: !1, error: t.error };
    case "RESET":
      return { value: void 0, error: void 0 };
    default:
      return p;
  }
}
function Mh() {
  return nt.useReducer(vR, { value: void 0, error: void 0 });
}
const Ty = typeof window < "u", IC = Ty && window.location.protocol === "file:";
function wR(p) {
  return typeof p < "u";
}
function Ga(p) {
  return wR(p) && p !== null;
}
function AR(p) {
  return typeof p == "string";
}
function SR(p) {
  return p instanceof ArrayBuffer;
}
function bR(p) {
  return ae(Ty), p instanceof Blob;
}
function rA(p) {
  return AR(p) && /^data:/.test(p);
}
function o1(p) {
  ae(rA(p));
  const [t = "", e = ""] = p.split(",");
  return t.split(";").indexOf("base64") !== -1 ? atob(e) : unescape(e);
}
function ER() {
  return Ty && window.devicePixelRatio || 1;
}
const FC = "On Chromium based browsers, you can use --allow-file-access-from-files flag for debugging purposes.";
function a1() {
  Dn(!IC, `Loading PDF as base64 strings/URLs may not work on protocols other than HTTP/HTTPS. ${FC}`);
}
function _R() {
  Dn(!IC, `Loading PDF.js worker may not work on protocols other than HTTP/HTTPS. ${FC}`);
}
function dc(p) {
  p != null && p.cancel && p.cancel();
}
function oA(p, t) {
  return Object.defineProperty(p, "width", {
    get() {
      return this.view[2] * t;
    },
    configurable: !0
  }), Object.defineProperty(p, "height", {
    get() {
      return this.view[3] * t;
    },
    configurable: !0
  }), Object.defineProperty(p, "originalWidth", {
    get() {
      return this.view[2];
    },
    configurable: !0
  }), Object.defineProperty(p, "originalHeight", {
    get() {
      return this.view[3];
    },
    configurable: !0
  }), p;
}
function CR(p) {
  return p.name === "RenderingCancelledException";
}
function xR(p) {
  return new Promise((t, e) => {
    const n = new FileReader();
    n.onload = () => {
      if (!n.result)
        return e(new Error("Error while reading a file."));
      t(n.result);
    }, n.onerror = (s) => {
      if (!s.target)
        return e(new Error("Error while reading a file."));
      const { error: a } = s.target;
      if (!a)
        return e(new Error("Error while reading a file."));
      switch (a.code) {
        case a.NOT_FOUND_ERR:
          return e(new Error("Error while reading a file: File not found."));
        case a.SECURITY_ERR:
          return e(new Error("Error while reading a file: Security error."));
        case a.ABORT_ERR:
          return e(new Error("Error while reading a file: Aborted."));
        default:
          return e(new Error("Error while reading a file."));
      }
    }, n.readAsArrayBuffer(p);
  });
}
const { PDFDataRangeTransport: TR } = XP, kR = (p, t) => {
  switch (t) {
    case r1.NEED_PASSWORD: {
      const e = prompt("Enter the password to open this PDF file.");
      p(e);
      break;
    }
    case r1.INCORRECT_PASSWORD: {
      const e = prompt("Invalid password. Please try again.");
      p(e);
      break;
    }
  }
};
function l1(p) {
  return typeof p == "object" && p !== null && ("data" in p || "range" in p || "url" in p);
}
const PR = nt.forwardRef(function({ children: t, className: e, error: n = "Failed to load PDF file.", externalLinkRel: s, externalLinkTarget: a, file: l, inputRef: u, imageResourcesPath: h, loading: f = "Loading PDF…", noData: g = "No PDF file specified.", onItemClick: y, onLoadError: v, onLoadProgress: S, onLoadSuccess: x, onPassword: _ = kR, onSourceError: k, onSourceSuccess: T, options: R, renderMode: P, rotate: L, scale: D, ...N }, B) {
  const [$, V] = Mh(), { value: X, error: q } = $, [it, lt] = Mh(), { value: rt, error: vt } = it, tt = nt.useRef(new yR()), Q = nt.useRef([]), W = nt.useRef(void 0), ot = nt.useRef(void 0);
  l && l !== W.current && l1(l) && (Dn(!eu(l, W.current), `File prop passed to <Document /> changed, but it's equal to previous one. This might result in unnecessary reloads. Consider memoizing the value passed to "file" prop.`), W.current = l), R && R !== ot.current && (Dn(!eu(R, ot.current), `Options prop passed to <Document /> changed, but it's equal to previous one. This might result in unnecessary reloads. Consider memoizing the value passed to "options" prop.`), ot.current = R);
  const st = nt.useRef({
    // Handling jumping to internal links target
    scrollPageIntoView: (Ht) => {
      const { dest: Jt, pageNumber: Te, pageIndex: Ie = Te - 1 } = Ht;
      if (y) {
        y({ dest: Jt, pageIndex: Ie, pageNumber: Te });
        return;
      }
      const Rs = Q.current[Ie];
      if (Rs) {
        Rs.scrollIntoView();
        return;
      }
      Dn(!1, `An internal link leading to page ${Te} was clicked, but neither <Document> was provided with onItemClick nor it was able to find the page within itself. Either provide onItemClick to <Document> and handle navigating by yourself or ensure that all pages are rendered within <Document>.`);
    }
  });
  nt.useImperativeHandle(B, () => ({
    linkService: tt,
    pages: Q,
    viewer: st
  }), []);
  function I() {
    T && T();
  }
  function z() {
    q && (Dn(!1, q.toString()), k && k(q));
  }
  function K() {
    V({ type: "RESET" });
  }
  nt.useEffect(K, [l, V]);
  const ct = nt.useCallback(async () => {
    if (!l)
      return null;
    if (typeof l == "string")
      return rA(l) ? { data: o1(l) } : (a1(), { url: l });
    if (l instanceof TR)
      return { range: l };
    if (SR(l))
      return { data: l };
    if (Ty && bR(l))
      return { data: await xR(l) };
    if (ae(typeof l == "object"), ae(l1(l)), "url" in l && typeof l.url == "string") {
      if (rA(l.url)) {
        const { url: Ht, ...Jt } = l;
        return { data: o1(Ht), ...Jt };
      }
      a1();
    }
    return l;
  }, [l]);
  nt.useEffect(() => {
    const Ht = Hp(ct());
    return Ht.promise.then((Jt) => {
      V({ type: "RESOLVE", value: Jt });
    }).catch((Jt) => {
      V({ type: "REJECT", error: Jt });
    }), () => {
      dc(Ht);
    };
  }, [ct, V]), nt.useEffect(() => {
    if (!(typeof X > "u")) {
      if (X === !1) {
        z();
        return;
      }
      I();
    }
  }, [X]);
  function Et() {
    rt && (x && x(rt), Q.current = new Array(rt.numPages), tt.current.setDocument(rt));
  }
  function xt() {
    vt && (Dn(!1, vt.toString()), v && v(vt));
  }
  nt.useEffect(function() {
    lt({ type: "RESET" });
  }, [lt, X]), nt.useEffect(function() {
    if (!X)
      return;
    const Jt = R ? { ...X, ...R } : X, Te = yA(Jt);
    S && (Te.onProgress = S), _ && (Te.onPassword = _);
    const Ie = Te, Rs = Ie.promise.then((ho) => {
      lt({ type: "RESOLVE", value: ho });
    }).catch((ho) => {
      Ie.destroyed || lt({ type: "REJECT", error: ho });
    });
    return () => {
      Rs.finally(() => Ie.destroy());
    };
  }, [R, lt, X]), nt.useEffect(() => {
    if (!(typeof rt > "u")) {
      if (rt === !1) {
        xt();
        return;
      }
      Et();
    }
  }, [rt]), nt.useEffect(function() {
    tt.current.setViewer(st.current), tt.current.setExternalLinkRel(s), tt.current.setExternalLinkTarget(a);
  }, [s, a]);
  const bt = nt.useCallback((Ht, Jt) => {
    Q.current[Ht] = Jt;
  }, []), Ot = nt.useCallback((Ht) => {
    delete Q.current[Ht];
  }, []), Bt = nt.useMemo(() => ({
    imageResourcesPath: h,
    linkService: tt.current,
    onItemClick: y,
    pdf: rt,
    registerPage: bt,
    renderMode: P,
    rotate: L,
    scale: D,
    unregisterPage: Ot
  }), [h, y, rt, bt, P, L, D, Ot]), Dt = nt.useMemo(
    () => LC(N, () => rt),
    // biome-ignore lint/correctness/useExhaustiveDependencies: FIXME
    [N, rt]
  );
  function Qe() {
    function Ht(Te) {
      return !!(Te != null && Te.pdf);
    }
    if (!Ht(Bt))
      throw new Error("pdf is undefined");
    const Jt = typeof t == "function" ? t(Bt) : t;
    return St.jsx(DC.Provider, { value: Bt, children: Jt });
  }
  function Di() {
    return l ? rt == null ? St.jsx(nu, { type: "loading", children: typeof f == "function" ? f() : f }) : rt === !1 ? St.jsx(nu, { type: "error", children: typeof n == "function" ? n() : n }) : Qe() : St.jsx(nu, { type: "no-data", children: typeof g == "function" ? g() : g });
  }
  return St.jsx("div", {
    className: xy("react-pdf__Document", e),
    // Assertion is needed for React 18 compatibility
    ref: u,
    ...Dt,
    children: Di()
  });
});
function NC() {
  return nt.useContext(DC);
}
function OC() {
  for (var p = [], t = 0; t < arguments.length; t++)
    p[t] = arguments[t];
  var e = p.filter(Boolean);
  if (e.length <= 1) {
    var n = e[0];
    return n || null;
  }
  return function(a) {
    for (var l = 0, u = e; l < u.length; l++) {
      var h = u[l];
      typeof h == "function" ? h(a) : h && (h.current = a);
    }
  };
}
const BC = nt.createContext(null);
function ky() {
  return nt.useContext(BC);
}
function RR() {
  const p = NC(), t = ky();
  ae(t);
  const e = { ...p, ...t }, { imageResourcesPath: n, linkService: s, onGetAnnotationsError: a, onGetAnnotationsSuccess: l, onRenderAnnotationLayerError: u, onRenderAnnotationLayerSuccess: h, page: f, pdf: g, renderForms: y, rotate: v, scale: S = 1 } = e;
  ae(g), ae(f), ae(s);
  const [x, _] = Mh(), { value: k, error: T } = x, R = nt.useRef(null);
  Dn(Number.parseInt(window.getComputedStyle(document.body).getPropertyValue("--react-pdf-annotation-layer"), 10) === 1, "AnnotationLayer styles not found. Read more: https://github.com/wojtekmaj/react-pdf#support-for-annotations");
  function P() {
    k && l && l(k);
  }
  function L() {
    T && (Dn(!1, T.toString()), a && a(T));
  }
  nt.useEffect(function() {
    _({ type: "RESET" });
  }, [_, f]), nt.useEffect(function() {
    if (!f)
      return;
    const V = Hp(f.getAnnotations()), X = V;
    return V.promise.then((q) => {
      _({ type: "RESOLVE", value: q });
    }).catch((q) => {
      _({ type: "REJECT", error: q });
    }), () => {
      dc(X);
    };
  }, [_, f]), nt.useEffect(() => {
    if (k !== void 0) {
      if (k === !1) {
        L();
        return;
      }
      P();
    }
  }, [k]);
  function D() {
    h && h();
  }
  function N($) {
    Dn(!1, `${$}`), u && u($);
  }
  const B = nt.useMemo(() => f.getViewport({ scale: S, rotation: v }), [f, v, S]);
  return nt.useEffect(function() {
    if (!g || !f || !s || !k)
      return;
    const { current: V } = R;
    if (!V)
      return;
    const X = B.clone({ dontFlip: !0 }), q = {
      accessibilityManager: null,
      // TODO: Implement this
      annotationCanvasMap: null,
      // TODO: Implement this
      annotationEditorUIManager: null,
      // TODO: Implement this
      annotationStorage: g.annotationStorage,
      commentManager: null,
      // TODO: Implement this
      div: V,
      l10n: null,
      // TODO: Implement this
      linkService: s,
      page: f,
      structTreeLayer: null,
      // TODO: Implement this
      viewport: X
    }, it = {
      annotations: k,
      annotationStorage: g.annotationStorage,
      div: V,
      imageResourcesPath: n,
      linkService: s,
      page: f,
      renderForms: y,
      viewport: X
    };
    V.innerHTML = "";
    try {
      new SA(q).render(it), D();
    } catch (lt) {
      N(lt);
    }
    return () => {
    };
  }, [k, n, s, f, g, y, B]), St.jsx("div", { className: xy("react-pdf__Page__annotations", "annotationLayer"), ref: R });
}
const zC = {
  // Document level structure types
  Document: null,
  // There's a "document" role, but it doesn't make sense here.
  DocumentFragment: null,
  // Grouping level structure types
  Part: "group",
  Sect: "group",
  // XXX: There's a "section" role, but it's abstract.
  Div: "group",
  Aside: "note",
  NonStruct: "none",
  // Block level structure types
  P: null,
  // H<n>,
  H: "heading",
  Title: null,
  FENote: "note",
  // Sub-block level structure type
  Sub: "group",
  // General inline level structure types
  Lbl: null,
  Span: null,
  Em: null,
  Strong: null,
  Link: "link",
  Annot: "note",
  Form: "form",
  // Ruby and Warichu structure types
  Ruby: null,
  RB: null,
  RT: null,
  RP: null,
  Warichu: null,
  WT: null,
  WP: null,
  // List standard structure types
  L: "list",
  LI: "listitem",
  LBody: null,
  // Table standard structure types
  Table: "table",
  TR: "row",
  TH: "columnheader",
  TD: "cell",
  THead: "columnheader",
  TBody: null,
  TFoot: null,
  // Standard structure type Caption
  Caption: null,
  // Standard structure type Figure
  Figure: "figure",
  // Standard structure type Formula
  Formula: null,
  // standard structure type Artifact
  Artifact: null
}, MR = /^H(\d+)$/;
function LR(p) {
  return p in zC;
}
function Py(p) {
  return "children" in p;
}
function UC(p) {
  return Py(p) ? p.children.length === 1 && 0 in p.children && "id" in p.children[0] : !1;
}
function DR(p) {
  const t = {};
  if (Py(p)) {
    const { role: e } = p, n = e.match(MR);
    if (n)
      t.role = "heading", t["aria-level"] = Number(n[1]);
    else if (LR(e)) {
      const s = zC[e];
      s && (t.role = s);
    }
  }
  return t;
}
function HC(p) {
  const t = {};
  if (Py(p)) {
    if (p.alt !== void 0 && (t["aria-label"] = p.alt), p.lang !== void 0 && (t.lang = p.lang), UC(p)) {
      const [e] = p.children;
      if (e) {
        const n = HC(e);
        return {
          ...t,
          ...n
        };
      }
    }
  } else "id" in p && (t["aria-owns"] = p.id);
  return t;
}
function IR(p) {
  return p ? {
    ...DR(p),
    ...HC(p)
  } : null;
}
function jC({ className: p, node: t }) {
  const e = nt.useMemo(() => IR(t), [t]), n = nt.useMemo(() => !Py(t) || UC(t) ? null : t.children.map((s, a) => (
    // biome-ignore lint/suspicious/noArrayIndexKey: index is stable here
    St.jsx(jC, { node: s }, a)
  )), [t]);
  return St.jsx("span", { className: p, ...e, children: n });
}
function FR() {
  const p = ky();
  ae(p);
  const { onGetStructTreeError: t, onGetStructTreeSuccess: e } = p, [n, s] = Mh(), { value: a, error: l } = n, { customTextRenderer: u, page: h } = p;
  function f() {
    a && e && e(a);
  }
  function g() {
    l && (Dn(!1, l.toString()), t && t(l));
  }
  return nt.useEffect(function() {
    s({ type: "RESET" });
  }, [s, h]), nt.useEffect(function() {
    if (u || !h)
      return;
    const v = Hp(h.getStructTree()), S = v;
    return v.promise.then((x) => {
      s({ type: "RESOLVE", value: x });
    }).catch((x) => {
      s({ type: "REJECT", error: x });
    }), () => dc(S);
  }, [u, h, s]), nt.useEffect(() => {
    if (a !== void 0) {
      if (a === !1) {
        g();
        return;
      }
      f();
    }
  }, [a]), a ? St.jsx(jC, { className: "react-pdf__Page__structTree structTree", node: a }) : null;
}
const c1 = lr;
function NR(p) {
  const t = ky();
  ae(t);
  const e = { ...t, ...p }, { _className: n, canvasBackground: s, devicePixelRatio: a = ER(), onRenderError: l, onRenderSuccess: u, page: h, renderForms: f, renderTextLayer: g, rotate: y, scale: v } = e, { canvasRef: S } = p;
  ae(h);
  const x = nt.useRef(null);
  function _() {
    h && u && u(oA(h, v));
  }
  function k(L) {
    CR(L) || (Dn(!1, L.toString()), l && l(L));
  }
  const T = nt.useMemo(() => h.getViewport({ scale: v * a, rotation: y }), [a, h, y, v]), R = nt.useMemo(() => h.getViewport({ scale: v, rotation: y }), [h, y, v]);
  nt.useEffect(function() {
    if (!h)
      return;
    h.cleanup();
    const { current: D } = x;
    if (!D)
      return;
    D.width = T.width, D.height = T.height, D.style.width = `${Math.floor(R.width)}px`, D.style.height = `${Math.floor(R.height)}px`, D.style.visibility = "hidden";
    const N = {
      annotationMode: f ? c1.ENABLE_FORMS : c1.ENABLE,
      canvas: D,
      canvasContext: D.getContext("2d", { alpha: !1 }),
      viewport: T
    };
    s && (N.background = s);
    const B = h.render(N), $ = B;
    return B.promise.then(() => {
      D.style.visibility = "", _();
    }).catch(k), () => dc($);
  }, [s, h, f, T, R]);
  const P = nt.useCallback(() => {
    const { current: L } = x;
    L && (L.width = 0, L.height = 0);
  }, []);
  return nt.useEffect(() => P, [P]), St.jsx("canvas", { className: `${n}__canvas`, dir: "ltr", ref: OC(S, x), style: {
    display: "block",
    userSelect: "none"
  }, children: g ? St.jsx(FR, {}) : null });
}
function OR(p) {
  return "str" in p;
}
function BR() {
  const p = ky();
  ae(p);
  const { customTextRenderer: t, onGetTextError: e, onGetTextSuccess: n, onRenderTextLayerError: s, onRenderTextLayerSuccess: a, page: l, pageIndex: u, pageNumber: h, rotate: f, scale: g } = p;
  ae(l);
  const [y, v] = Mh(), { value: S, error: x } = y, _ = nt.useRef(null);
  Dn(Number.parseInt(window.getComputedStyle(document.body).getPropertyValue("--react-pdf-text-layer"), 10) === 1, "TextLayer styles not found. Read more: https://github.com/wojtekmaj/react-pdf#support-for-text-layer");
  function k() {
    S && n && n(S);
  }
  function T() {
    x && (Dn(!1, x.toString()), e && e(x));
  }
  nt.useEffect(function() {
    v({ type: "RESET" });
  }, [l, v]), nt.useEffect(function() {
    if (!l)
      return;
    const $ = Hp(l.getTextContent()), V = $;
    return $.promise.then((X) => {
      v({ type: "RESOLVE", value: X });
    }).catch((X) => {
      v({ type: "REJECT", error: X });
    }), () => dc(V);
  }, [l, v]), nt.useEffect(() => {
    if (S !== void 0) {
      if (S === !1) {
        T();
        return;
      }
      k();
    }
  }, [S]);
  const R = nt.useCallback(() => {
    a && a();
  }, [a]), P = nt.useCallback((B) => {
    Dn(!1, B.toString()), s && s(B);
  }, [s]);
  function L() {
    const B = _.current;
    B && B.classList.add("selecting");
  }
  function D() {
    const B = _.current;
    B && B.classList.remove("selecting");
  }
  const N = nt.useMemo(() => l.getViewport({ scale: g, rotation: f }), [l, f, g]);
  return nt.useLayoutEffect(function() {
    if (!l || !S)
      return;
    const { current: $ } = _;
    if (!$)
      return;
    $.innerHTML = "";
    const V = l.streamTextContent({ includeMarkedContent: !0 }), X = {
      container: $,
      textContentSource: V,
      viewport: N
    }, q = new Kd(X), it = q;
    return q.render().then(() => {
      const lt = document.createElement("div");
      lt.className = "endOfContent", $.append(lt);
      const rt = $.querySelectorAll('[role="presentation"]');
      if (t) {
        let vt = 0;
        S.items.forEach((tt, Q) => {
          if (!OR(tt))
            return;
          const W = rt[vt];
          if (!W)
            return;
          const ot = t({
            pageIndex: u,
            pageNumber: h,
            itemIndex: Q,
            ...tt
          });
          W.innerHTML = ot, vt += tt.str && tt.hasEOL ? 2 : 1;
        });
      }
      R();
    }).catch(P), () => dc(it);
  }, [
    t,
    P,
    R,
    l,
    u,
    h,
    S,
    N
  ]), // biome-ignore lint/a11y/noStaticElementInteractions: False positive caused by non interactive wrapper listening for bubbling events
  St.jsx("div", { className: xy("react-pdf__Page__textContent", "textLayer"), onMouseUp: D, onMouseDown: L, ref: _ });
}
const u1 = 1;
function zR(p) {
  const e = { ...NC(), ...p }, { _className: n = "react-pdf__Page", _enableRegisterUnregisterPage: s = !0, canvasBackground: a, canvasRef: l, children: u, className: h, customRenderer: f, customTextRenderer: g, devicePixelRatio: y, error: v = "Failed to load the page.", height: S, inputRef: x, loading: _ = "Loading page…", noData: k = "No page specified.", onGetAnnotationsError: T, onGetAnnotationsSuccess: R, onGetStructTreeError: P, onGetStructTreeSuccess: L, onGetTextError: D, onGetTextSuccess: N, onLoadError: B, onLoadSuccess: $, onRenderAnnotationLayerError: V, onRenderAnnotationLayerSuccess: X, onRenderError: q, onRenderSuccess: it, onRenderTextLayerError: lt, onRenderTextLayerSuccess: rt, pageIndex: vt, pageNumber: tt, pdf: Q, registerPage: W, renderAnnotationLayer: ot = !0, renderForms: st = !1, renderMode: I = "canvas", renderTextLayer: z = !0, rotate: K, scale: ct = u1, unregisterPage: Et, width: xt, ...bt } = e, [Ot, Bt] = Mh(), { value: Dt, error: Qe } = Ot, Di = nt.useRef(null);
  ae(Q);
  const Ht = Ga(tt) ? tt - 1 : vt ?? null, Jt = tt ?? (Ga(vt) ? vt + 1 : null), Te = K ?? (Dt ? Dt.rotate : null), Ie = nt.useMemo(() => {
    if (!Dt)
      return null;
    let Ii = 1;
    const ss = ct ?? u1;
    if (xt || S) {
      const rs = Dt.getViewport({ scale: 1, rotation: Te });
      xt ? Ii = xt / rs.width : S && (Ii = S / rs.height);
    }
    return ss * Ii;
  }, [S, Dt, Te, ct, xt]);
  nt.useEffect(function() {
    return () => {
      Ga(Ht) && s && Et && Et(Ht);
    };
  }, [s, Q, Ht, Et]);
  function Rs() {
    if ($) {
      if (!Dt || !Ie)
        return;
      $(oA(Dt, Ie));
    }
    if (s && W) {
      if (!Ga(Ht) || !Di.current)
        return;
      W(Ht, Di.current);
    }
  }
  function ho() {
    Qe && (Dn(!1, Qe.toString()), B && B(Qe));
  }
  nt.useEffect(function() {
    Bt({ type: "RESET" });
  }, [Bt, Q, Ht]), nt.useEffect(function() {
    if (!Q || !Jt)
      return;
    const ss = Hp(Q.getPage(Jt)), rs = ss;
    return ss.promise.then((hr) => {
      Bt({ type: "RESOLVE", value: hr });
    }).catch((hr) => {
      Bt({ type: "REJECT", error: hr });
    }), () => dc(rs);
  }, [Bt, Q, Jt]), nt.useEffect(() => {
    if (Dt !== void 0) {
      if (Dt === !1) {
        ho();
        return;
      }
      Rs();
    }
  }, [Dt, Ie]);
  const Ma = nt.useMemo(() => (
    // Technically there cannot be page without pageIndex, pageNumber, rotate and scale, but TypeScript doesn't know that
    Ga(Ht) && Jt && Ga(Te) && Ga(Ie) ? {
      _className: n,
      canvasBackground: a,
      customTextRenderer: g,
      devicePixelRatio: y,
      onGetAnnotationsError: T,
      onGetAnnotationsSuccess: R,
      onGetStructTreeError: P,
      onGetStructTreeSuccess: L,
      onGetTextError: D,
      onGetTextSuccess: N,
      onRenderAnnotationLayerError: V,
      onRenderAnnotationLayerSuccess: X,
      onRenderError: q,
      onRenderSuccess: it,
      onRenderTextLayerError: lt,
      onRenderTextLayerSuccess: rt,
      page: Dt,
      pageIndex: Ht,
      pageNumber: Jt,
      renderForms: st,
      renderTextLayer: z,
      rotate: Te,
      scale: Ie
    } : null
  ), [
    n,
    a,
    g,
    y,
    T,
    R,
    P,
    L,
    D,
    N,
    V,
    X,
    q,
    it,
    lt,
    rt,
    Dt,
    Ht,
    Jt,
    st,
    z,
    Te,
    Ie
  ]), La = nt.useMemo(
    () => LC(bt, () => Dt && (Ie ? oA(Dt, Ie) : void 0)),
    // biome-ignore lint/correctness/useExhaustiveDependencies: FIXME
    [bt, Dt, Ie]
  ), is = `${Ht}@${Ie}/${Te}`;
  function Ih() {
    switch (I) {
      case "custom":
        return ae(f), St.jsx(f, {}, `${is}_custom`);
      case "none":
        return null;
      case "canvas":
      default:
        return St.jsx(NR, { canvasRef: l }, `${is}_canvas`);
    }
  }
  function jp() {
    return z ? St.jsx(BR, {}, `${is}_text`) : null;
  }
  function $p() {
    return ot ? St.jsx(RR, {}, `${is}_annotations`) : null;
  }
  function Vp() {
    function Ii(rs) {
      return !!(rs != null && rs.page);
    }
    if (!Ii(Ma))
      throw new Error("page is undefined");
    const ss = typeof u == "function" ? u(Ma) : u;
    return St.jsxs(BC.Provider, { value: Ma, children: [Ih(), jp(), $p(), ss] });
  }
  function Wp() {
    return Jt ? Q === null || Dt === void 0 || Dt === null ? St.jsx(nu, { type: "loading", children: typeof _ == "function" ? _() : _ }) : Q === !1 || Dt === !1 ? St.jsx(nu, { type: "error", children: typeof v == "function" ? v() : v }) : Vp() : St.jsx(nu, { type: "no-data", children: typeof k == "function" ? k() : k });
  }
  return St.jsx("div", {
    className: xy(n, h),
    "data-page-number": Jt,
    // Assertion is needed for React 18 compatibility
    ref: OC(x, Di),
    style: {
      "--scale-round-x": "1px",
      "--scale-round-y": "1px",
      "--scale-factor": "1",
      "--user-unit": `${Ie}`,
      "--total-scale-factor": "calc(var(--scale-factor) * var(--user-unit))",
      backgroundColor: a || "white",
      position: "relative",
      minWidth: "min-content",
      minHeight: "min-content"
    },
    ...La,
    children: Wp()
  });
}
_R();
ts.workerSrc = "pdf.worker.mjs";
const UR = "_container_dbb9e_18", HR = "_content_dbb9e_25", jR = "_noFileContainer_dbb9e_33", $R = "_noFileIcon_dbb9e_44", VR = "_noFileTitle_dbb9e_50", WR = "_noFileText_dbb9e_57", GR = "_loadingContainer_dbb9e_63", XR = "_loadingContent_dbb9e_70", qR = "_spinner_dbb9e_74", YR = "_loadingText_dbb9e_93", KR = "_errorContainer_dbb9e_99", QR = "_errorIcon_dbb9e_107", ZR = "_errorTitle_dbb9e_113", JR = "_errorText_dbb9e_120", tM = "_documentContainer_dbb9e_126", eM = "_virtualContainer_dbb9e_133", nM = "_virtualItem_dbb9e_138", iM = "_pageContainer_dbb9e_145", sM = "_pdfPage_dbb9e_154", rM = "_pagePlaceholder_dbb9e_162", oM = "_zoomControls_dbb9e_174", aM = "_visible_dbb9e_196", lM = "_zoomButton_dbb9e_205", cM = "_disabled_dbb9e_245", he = {
  container: UR,
  content: HR,
  noFileContainer: jR,
  noFileIcon: $R,
  noFileTitle: VR,
  noFileText: WR,
  loadingContainer: GR,
  loadingContent: XR,
  spinner: qR,
  loadingText: YR,
  errorContainer: KR,
  errorIcon: QR,
  errorTitle: ZR,
  errorText: JR,
  documentContainer: tM,
  virtualContainer: eM,
  virtualItem: nM,
  pageContainer: iM,
  pdfPage: sM,
  pagePlaceholder: rM,
  zoomControls: oM,
  visible: aM,
  zoomButton: lM,
  disabled: cM
}, uM = "/media/", hM = () => {
  var p;
  if (typeof window > "u")
    return null;
  if ((p = window == null ? void 0 : window.__streamlit) != null && p.DOWNLOAD_ASSETS_BASE_URL)
    return window.__streamlit.DOWNLOAD_ASSETS_BASE_URL;
  try {
    const t = new URL(window.location.href);
    return `${t.origin}${t.pathname}`;
  } catch {
    return null;
  }
}, dM = (p) => {
  if (!p)
    return;
  const t = p.replace(/^\/+/, "/");
  if (!t.startsWith(uM))
    return p;
  const e = hM();
  if (!e)
    return p;
  try {
    const n = new URL(e), s = n.pathname || "/";
    let a;
    if (s.endsWith("/"))
      a = s;
    else {
      const f = s.lastIndexOf("/");
      a = s.slice(0, Math.max(0, f + 1)), a || (a = "/");
    }
    a = a.replace(/\/{2,}/g, "/"), a.startsWith("/") || (a = `/${a}`), a !== "/" && !a.endsWith("/") && (a = `${a}/`);
    const l = n.origin, u = t.replace(/^\/+/, "");
    return `${l}${a}${u}`;
  } catch {
    const n = e.replace(/\/+$/, ""), s = t.replace(/^\/+/, "");
    return `${n}/${s}`;
  }
};
ts.workerSrc = new URL(
  /* @vite-ignore */
  "./workers/pdf.worker.min.mjs",
  import.meta.url
).toString();
const h1 = 0.5, d1 = 3, f1 = 0.25, fM = 5e3, pM = 800, D0 = 12;
function gM({
  file: p,
  height: t = 600
}) {
  const e = dM(p), [n, s] = nt.useState(0), [a, l] = nt.useState(!0), [u, h] = nt.useState(
    null
  ), [f, g] = nt.useState(1), [y, v] = nt.useState(!1), [S, x] = nt.useState({}), [_, k] = nt.useState({}), [T, R] = nt.useState(0), P = nt.useRef(null), L = nt.useRef(null), D = nt.useRef(
    null
  ), N = nt.useRef(null), B = nt.useRef(!1), $ = nt.useMemo(
    () => ({
      cMapUrl: new URL(
        /* @vite-ignore */
        "./cmaps/",
        import.meta.url
      ).toString(),
      cMapPacked: !0,
      standardFontDataUrl: new URL(
        /* @vite-ignore */
        "./standard_fonts/",
        import.meta.url
      ).toString()
    }),
    []
  ), V = nt.useMemo(() => {
    const I = Object.values(S);
    if (I.length === 0)
      return pM;
    const z = I.reduce(
      (K, ct) => K + ct,
      0
    );
    return Math.round(z / I.length);
  }, [S]), X = nt.useCallback(() => P.current, []), q = nt.useCallback(
    (I) => {
      const z = I.getAttribute("data-index"), K = z ? !!S[parseInt(z, 10)] : !1;
      return I && K ? I.getBoundingClientRect().height + D0 : V * f + D0;
    },
    [S, f, V]
  ), it = ak({
    count: n,
    getScrollElement: X,
    estimateSize: nt.useCallback(
      (I) => (S[I] || V) * f + D0,
      [S, f, V]
    ),
    overscan: 3,
    // Render 3 pages above and below viewport
    measureElement: q
  }), lt = nt.useCallback(() => {
    P.current && R(P.current.clientWidth);
  }, []);
  nt.useEffect(() => {
    const I = new ResizeObserver(lt);
    return P.current && I.observe(P.current), () => {
      I.disconnect();
    };
  }, [lt]), nt.useEffect(() => {
    const I = () => {
      v(!1), D.current && (clearTimeout(D.current), D.current = null);
    }, z = P.current;
    return z && z.addEventListener("scroll", I, { passive: !0 }), () => {
      z && z.removeEventListener("scroll", I);
    };
  }, []), nt.useEffect(() => {
    const I = () => {
      D.current && clearTimeout(D.current), D.current = setTimeout(() => {
        v(!1);
      }, fM);
    }, z = () => {
      y && I();
    }, K = N.current;
    return K && y && (K.addEventListener("mousemove", z), I()), () => {
      K && K.removeEventListener("mousemove", z), D.current && clearTimeout(D.current);
    };
  }, [y]);
  const rt = nt.useCallback(
    ({ numPages: I }) => {
      s(I), l(!1), h(null);
    },
    []
  ), vt = nt.useCallback((I) => {
    l(!1);
    let z = I.message.toLowerCase(), K = "", ct = "";
    z.includes("cors") || z.includes("cross-origin") ? (K = "Unable to load PDF from external source", ct = "The PDF is hosted on a different domain that doesn't allow cross-origin requests. Try downloading the PDF and uploading it directly, or ask the website owner to enable CORS.") : z.includes("invalid pdf") || z.includes("pdf header") || z.includes("not a pdf") ? (K = "Invalid PDF file", ct = "The file doesn't appear to be a valid PDF. Please check that the file is not corrupted and is actually a PDF document.") : z.includes("network") || z.includes("fetch") || z.includes("load") ? (K = "Network error loading PDF", ct = "Unable to download the PDF. This could be due to network issues or CORS restrictions. Please check your internet connection.") : z.includes("not found") || z.includes("404") ? (K = "PDF not found", ct = "The PDF file could not be found at the specified location. Please check the URL or file path.") : z.includes("unauthorized") || z.includes("401") || z.includes("403") ? (K = "Access denied", ct = "You don't have permission to access this PDF.") : (K = "Unable to load PDF", ct = `Error details: ${I.message}`), h({ main: K, help: ct });
  }, []), tt = nt.useCallback(
    (I, z) => {
      x((K) => ({
        ...K,
        [I]: z.height
      })), k((K) => ({
        ...K,
        [I]: z.width
      })), it.measureElement(it.scrollElement);
    },
    [it]
  ), Q = nt.useCallback(() => {
    I0.flushSync(() => {
      g((I) => Math.min(d1, I + f1));
    }), it.measure();
  }, [it]), W = nt.useCallback(() => {
    I0.flushSync(() => {
      g((I) => Math.max(h1, I - f1));
    }), it.measure();
  }, [it]), ot = f < d1, st = f > h1;
  return e ? /* @__PURE__ */ St.jsxs(
    "div",
    {
      ref: N,
      className: he.container,
      "data-testid": "pdf-container",
      style: {
        "--default-page-height": `${V}px`
      },
      onMouseEnter: () => {
        B.current = !0, v(!0);
      },
      onMouseLeave: () => {
        B.current = !1, v(!1), D.current && (clearTimeout(D.current), D.current = null);
      },
      children: [
        !a && !u && e && /* @__PURE__ */ St.jsxs(
          "div",
          {
            className: `${he.zoomControls} ${y ? he.visible : ""}`,
            children: [
              /* @__PURE__ */ St.jsx(
                "button",
                {
                  className: `${he.zoomButton} ${ot ? "" : he.disabled}`,
                  onClick: Q,
                  disabled: !ot,
                  title: "Zoom In",
                  children: /* @__PURE__ */ St.jsx(
                    "svg",
                    {
                      xmlns: "http://www.w3.org/2000/svg",
                      viewBox: "0 0 29 29",
                      fill: "currentColor",
                      children: /* @__PURE__ */ St.jsx("path", { d: "M14.5 8.5c-.75 0-1.5.75-1.5 1.5v3h-3c-.75 0-1.5.75-1.5 1.5S9.25 16 10 16h3v3c0 .75.75 1.5 1.5 1.5S16 19.75 16 19v-3h3c.75 0 1.5-.75 1.5-1.5S19.75 13 19 13h-3v-3c0-.75-.75-1.5-1.5-1.5z" })
                    }
                  )
                }
              ),
              /* @__PURE__ */ St.jsx(
                "button",
                {
                  className: `${he.zoomButton} ${st ? "" : he.disabled}`,
                  onClick: W,
                  disabled: !st,
                  title: "Zoom Out",
                  children: /* @__PURE__ */ St.jsx(
                    "svg",
                    {
                      xmlns: "http://www.w3.org/2000/svg",
                      viewBox: "0 0 29 29",
                      fill: "currentColor",
                      children: /* @__PURE__ */ St.jsx("path", { d: "M10 13c-.75 0-1.5.75-1.5 1.5S9.25 16 10 16h9c.75 0 1.5-.75 1.5-1.5S19.75 13 19 13h-9z" })
                    }
                  )
                }
              )
            ]
          }
        ),
        /* @__PURE__ */ St.jsxs(
          "div",
          {
            ref: P,
            className: he.content,
            style: { height: t },
            "data-testid": "pdf-content",
            children: [
              a && /* @__PURE__ */ St.jsx("div", { className: he.loadingContainer, "data-testid": "pdf-loading", children: /* @__PURE__ */ St.jsxs("div", { className: he.loadingContent, children: [
                /* @__PURE__ */ St.jsx("div", { className: he.spinner }),
                /* @__PURE__ */ St.jsx("p", { className: he.loadingText, children: "Loading PDF..." })
              ] }) }),
              u && /* @__PURE__ */ St.jsx("div", { className: he.errorContainer, "data-testid": "pdf-error", children: /* @__PURE__ */ St.jsxs("div", { children: [
                /* @__PURE__ */ St.jsx("div", { className: he.errorIcon, children: "⚠️" }),
                /* @__PURE__ */ St.jsx("h3", { className: he.errorTitle, children: u.main || "Failed to load PDF" }),
                /* @__PURE__ */ St.jsx("p", { className: he.errorText, children: u.help })
              ] }) }),
              !u && /* @__PURE__ */ St.jsx(
                "div",
                {
                  ref: L,
                  className: he.documentContainer,
                  "data-testid": "pdf-document-container",
                  children: /* @__PURE__ */ St.jsx(
                    PR,
                    {
                      file: e,
                      onLoadSuccess: rt,
                      onLoadError: vt,
                      loading: "",
                      error: "",
                      options: $,
                      children: n > 0 && /* @__PURE__ */ St.jsx(
                        "div",
                        {
                          className: he.virtualContainer,
                          style: {
                            height: `${it.getTotalSize()}px`
                          },
                          children: it.getVirtualItems().map((I) => {
                            const z = I.index, K = z + 1, ct = _[z] || 612 * f, Et = Math.max(
                              16,
                              (T - ct) / 2
                            );
                            return /* @__PURE__ */ St.jsx(
                              "div",
                              {
                                "data-index": z,
                                ref: it.measureElement,
                                className: he.virtualItem,
                                style: {
                                  left: `${Et}px`,
                                  transform: `translateY(${I.start}px)`,
                                  width: `${ct}px`
                                },
                                children: /* @__PURE__ */ St.jsx("div", { className: he.pageContainer, children: /* @__PURE__ */ St.jsx(
                                  zR,
                                  {
                                    pageNumber: K,
                                    scale: f,
                                    renderTextLayer: !1,
                                    renderAnnotationLayer: !1,
                                    className: he.pdfPage,
                                    onLoadSuccess: (xt) => tt(z, xt),
                                    loading: /* @__PURE__ */ St.jsx("div", { className: he.pagePlaceholder, children: /* @__PURE__ */ St.jsxs("p", { children: [
                                      "Loading page ",
                                      K,
                                      "..."
                                    ] }) })
                                  }
                                ) })
                              },
                              I.key
                            );
                          })
                        }
                      )
                    }
                  )
                }
              )
            ]
          }
        )
      ]
    }
  ) : /* @__PURE__ */ St.jsx(
    "div",
    {
      className: he.noFileContainer,
      "data-testid": "pdf-no-file-container",
      children: /* @__PURE__ */ St.jsxs("div", { children: [
        /* @__PURE__ */ St.jsx("div", { className: he.noFileIcon, children: "📄" }),
        /* @__PURE__ */ St.jsx("h3", { className: he.noFileTitle, children: "No PDF file provided" }),
        /* @__PURE__ */ St.jsx("p", { className: he.noFileText, children: "Please provide a PDF file to display" })
      ] })
    }
  );
}
const mM = (p) => {
  if (p === "stretch")
    return "100%";
  if (typeof p == "number" || typeof p == "string")
    return `${p}px`;
}, um = /* @__PURE__ */ new WeakMap(), wM = (p) => {
  const { data: t, parentElement: e } = p, n = e.querySelector(".react-root");
  if (!n)
    throw new Error("Unexpected: React root element not found");
  let s = um.get(e);
  return s || (s = YT.createRoot(n), um.set(e, s)), s.render(
    /* @__PURE__ */ St.jsx(nt.StrictMode, { children: /* @__PURE__ */ St.jsx(gM, { file: t == null ? void 0 : t.file, height: mM(t == null ? void 0 : t.height) }) })
  ), () => {
    const a = um.get(e);
    a && (a.unmount(), um.delete(e));
  };
};
export {
  wM as default
};
