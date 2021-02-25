var OpenSeadragon = OpenSeadragon || {}; OpenSeadragon["Annotations"] =
/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 3);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/*!********************************!*\
  !*** external "OpenSeadragon" ***!
  \********************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = OpenSeadragon;

/***/ }),
/* 1 */
/*!********************************************!*\
  !*** ./node_modules/preact/dist/preact.js ***!
  \********************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

!function(global, factory) {
     true ? factory(exports) : 'function' == typeof define && define.amd ? define([ 'exports' ], factory) : factory(global.preact = global.preact || {});
}(this, function(exports) {
    function VNode(nodeName, attributes, children) {
        this.nodeName = nodeName;
        this.attributes = attributes;
        this.children = children;
        this.key = attributes && attributes.key;
    }
    function extend(obj, props) {
        if (props) for (var i in props) if (void 0 !== props[i]) obj[i] = props[i];
        return obj;
    }
    function clone(obj) {
        return extend({}, obj);
    }
    function delve(obj, key) {
        for (var p = key.split('.'), i = 0; i < p.length && obj; i++) obj = obj[p[i]];
        return obj;
    }
    function toArray(obj, offset) {
        return [].slice.call(obj, offset);
    }
    function isFunction(obj) {
        return 'function' == typeof obj;
    }
    function isString(obj) {
        return 'string' == typeof obj;
    }
    function empty(x) {
        return void 0 === x || null === x;
    }
    function falsey(value) {
        return value === !1 || empty(value);
    }
    function hashToClassName(c) {
        var str = '';
        for (var prop in c) if (c[prop]) {
            if (str) str += ' ';
            str += prop;
        }
        return str;
    }
    function h(nodeName, attributes, firstChild) {
        var children, arr, lastSimple, len = arguments.length;
        if (len > 2) {
            var type = typeof firstChild;
            if (3 === len && 'object' !== type && 'function' !== type) {
                if (!falsey(firstChild)) children = [ String(firstChild) ];
            } else {
                children = [];
                for (var i = 2; i < len; i++) {
                    var _p = arguments[i];
                    if (!falsey(_p)) {
                        if (_p.join) arr = _p; else (arr = SHARED_TEMP_ARRAY)[0] = _p;
                        for (var j = 0; j < arr.length; j++) {
                            var child = arr[j], simple = !(falsey(child) || isFunction(child) || child instanceof VNode);
                            if (simple && !isString(child)) child = String(child);
                            if (simple && lastSimple) children[children.length - 1] += child; else if (!falsey(child)) {
                                children.push(child);
                                lastSimple = simple;
                            }
                        }
                    } else ;
                }
            }
        } else if (attributes && attributes.children) return h(nodeName, attributes, attributes.children);
        if (attributes) {
            if (attributes.children) delete attributes.children;
            if (!isFunction(nodeName)) {
                if ('className' in attributes) {
                    attributes.class = attributes.className;
                    delete attributes.className;
                }
                lastSimple = attributes.class;
                if (lastSimple && !isString(lastSimple)) attributes.class = hashToClassName(lastSimple);
            }
        }
        var p = new VNode(nodeName, attributes || void 0, children);
        if (options.vnode) options.vnode(p);
        return p;
    }
    function cloneElement(vnode, props) {
        return h(vnode.nodeName, extend(clone(vnode.attributes), props), arguments.length > 2 ? toArray(arguments, 2) : vnode.children);
    }
    function createLinkedState(component, key, eventPath) {
        var path = key.split('.'), p0 = path[0];
        return function(e) {
            var _component$setState;
            var v, i, t = e && e.currentTarget || this, s = component.state, obj = s;
            if (isString(eventPath)) {
                v = delve(e, eventPath);
                if (empty(v) && (t = t._component)) v = delve(t, eventPath);
            } else v = t.nodeName ? (t.nodeName + t.type).match(/^input(check|rad)/i) ? t.checked : t.value : e;
            if (isFunction(v)) v = v.call(t);
            if (path.length > 1) {
                for (i = 0; i < path.length - 1; i++) obj = obj[path[i]] || (obj[path[i]] = {});
                obj[path[i]] = v;
                v = s[p0];
            }
            component.setState((_component$setState = {}, _component$setState[p0] = v, _component$setState));
        };
    }
    function enqueueRender(component) {
        if (1 === items.push(component)) (options.debounceRendering || defer)(rerender);
    }
    function rerender() {
        if (items.length) {
            var p, currentItems = items;
            items = itemsOffline;
            itemsOffline = currentItems;
            while (p = currentItems.pop()) if (p._dirty) renderComponent(p);
        }
    }
    function isFunctionalComponent(vnode) {
        var nodeName = vnode && vnode.nodeName;
        return nodeName && isFunction(nodeName) && !(nodeName.prototype && nodeName.prototype.render);
    }
    function buildFunctionalComponent(vnode, context) {
        return vnode.nodeName(getNodeProps(vnode), context || EMPTY);
    }
    function ensureNodeData(node, data) {
        return node[ATTR_KEY] || (node[ATTR_KEY] = data || {});
    }
    function getNodeType(node) {
        if (node instanceof Text) return 3;
        if (node instanceof Element) return 1; else return 0;
    }
    function removeNode(node) {
        var p = node.parentNode;
        if (p) p.removeChild(node);
    }
    function setAccessor(node, name, value, old, isSvg) {
        ensureNodeData(node)[name] = value;
        if ('key' !== name && 'children' !== name && 'innerHTML' !== name) if ('class' === name && !isSvg) node.className = value || ''; else if ('style' === name) {
            if (!value || isString(value) || isString(old)) node.style.cssText = value || '';
            if (value && 'object' == typeof value) {
                if (!isString(old)) for (var i in old) if (!(i in value)) node.style[i] = '';
                for (var i in value) node.style[i] = 'number' == typeof value[i] && !NON_DIMENSION_PROPS[i] ? value[i] + 'px' : value[i];
            }
        } else if ('dangerouslySetInnerHTML' === name) {
            if (value) node.innerHTML = value.__html;
        } else if (name.match(/^on/i)) {
            var l = node._listeners || (node._listeners = {});
            name = toLowerCase(name.substring(2));
            if (value) {
                if (!l[name]) node.addEventListener(name, eventProxy);
            } else if (l[name]) node.removeEventListener(name, eventProxy);
            l[name] = value;
        } else if ('type' !== name && !isSvg && name in node) {
            setProperty(node, name, empty(value) ? '' : value);
            if (falsey(value)) node.removeAttribute(name);
        } else {
            var ns = isSvg && name.match(/^xlink\:?(.+)/);
            if (falsey(value)) if (ns) node.removeAttributeNS('http://www.w3.org/1999/xlink', toLowerCase(ns[1])); else node.removeAttribute(name); else if ('object' != typeof value && !isFunction(value)) if (ns) node.setAttributeNS('http://www.w3.org/1999/xlink', toLowerCase(ns[1]), value); else node.setAttribute(name, value);
        }
    }
    function setProperty(node, name, value) {
        try {
            node[name] = value;
        } catch (e) {}
    }
    function eventProxy(e) {
        return this._listeners[e.type](options.event && options.event(e) || e);
    }
    function getRawNodeAttributes(node) {
        var attrs = {};
        for (var i = node.attributes.length; i--; ) attrs[node.attributes[i].name] = node.attributes[i].value;
        return attrs;
    }
    function isSameNodeType(node, vnode) {
        if (isString(vnode)) return 3 === getNodeType(node);
        if (isString(vnode.nodeName)) return isNamedNode(node, vnode.nodeName);
        if (isFunction(vnode.nodeName)) return node._componentConstructor === vnode.nodeName || isFunctionalComponent(vnode); else ;
    }
    function isNamedNode(node, nodeName) {
        return node.normalizedNodeName === nodeName || toLowerCase(node.nodeName) === toLowerCase(nodeName);
    }
    function getNodeProps(vnode) {
        var defaultProps = vnode.nodeName.defaultProps, props = clone(defaultProps || vnode.attributes);
        if (defaultProps) extend(props, vnode.attributes);
        if (vnode.children) props.children = vnode.children;
        return props;
    }
    function collectNode(node) {
        cleanNode(node);
        var name = toLowerCase(node.nodeName), list = nodes[name];
        if (list) list.push(node); else nodes[name] = [ node ];
    }
    function createNode(nodeName, isSvg) {
        var name = toLowerCase(nodeName), node = nodes[name] && nodes[name].pop() || (isSvg ? document.createElementNS('http://www.w3.org/2000/svg', nodeName) : document.createElement(nodeName));
        ensureNodeData(node);
        node.normalizedNodeName = name;
        return node;
    }
    function cleanNode(node) {
        removeNode(node);
        if (1 === getNodeType(node)) {
            ensureNodeData(node, getRawNodeAttributes(node));
            node._component = node._componentConstructor = null;
        }
    }
    function flushMounts() {
        var c;
        while (c = mounts.pop()) if (c.componentDidMount) c.componentDidMount();
    }
    function diff(dom, vnode, context, mountAll, parent, rootComponent, nextSibling) {
        diffLevel++;
        var ret = idiff(dom, vnode, context, mountAll, rootComponent);
        if (parent && ret.parentNode !== parent) parent.insertBefore(ret, nextSibling || null);
        if (!--diffLevel) flushMounts();
        return ret;
    }
    function idiff(dom, vnode, context, mountAll, rootComponent) {
        var originalAttributes = vnode && vnode.attributes;
        while (isFunctionalComponent(vnode)) vnode = buildFunctionalComponent(vnode, context);
        if (empty(vnode)) {
            vnode = '';
            if (rootComponent) {
                if (dom) {
                    if (8 === dom.nodeType) return dom;
                    collectNode(dom);
                }
                return document.createComment(vnode);
            }
        }
        if (isString(vnode)) {
            if (dom) {
                if (3 === getNodeType(dom) && dom.parentNode) {
                    dom.nodeValue = vnode;
                    return dom;
                }
                collectNode(dom);
            }
            return document.createTextNode(vnode);
        }
        var svgMode, out = dom, nodeName = vnode.nodeName;
        if (isFunction(nodeName)) return buildComponentFromVNode(dom, vnode, context, mountAll);
        if (!isString(nodeName)) nodeName = String(nodeName);
        svgMode = 'svg' === toLowerCase(nodeName);
        if (svgMode) isSvgMode = !0;
        if (!dom) out = createNode(nodeName, isSvgMode); else if (!isNamedNode(dom, nodeName)) {
            out = createNode(nodeName, isSvgMode);
            while (dom.firstChild) out.appendChild(dom.firstChild);
            recollectNodeTree(dom);
        }
        if (vnode.children && 1 === vnode.children.length && 'string' == typeof vnode.children[0] && 1 === out.childNodes.length && out.firstChild instanceof Text) out.firstChild.nodeValue = vnode.children[0]; else if (vnode.children || out.firstChild) innerDiffNode(out, vnode.children, context, mountAll);
        diffAttributes(out, vnode.attributes);
        if (originalAttributes && originalAttributes.ref) (out[ATTR_KEY].ref = originalAttributes.ref)(out);
        if (svgMode) isSvgMode = !1;
        return out;
    }
    function innerDiffNode(dom, vchildren, context, mountAll) {
        var j, c, vchild, child, originalChildren = dom.childNodes, children = [], keyed = {}, keyedLen = 0, min = 0, len = originalChildren.length, childrenLen = 0, vlen = vchildren && vchildren.length;
        if (len) for (var i = 0; i < len; i++) {
            var _child = originalChildren[i], key = vlen ? (c = _child._component) ? c.__key : (c = _child[ATTR_KEY]) ? c.key : null : null;
            if (key || 0 === key) {
                keyedLen++;
                keyed[key] = _child;
            } else children[childrenLen++] = _child;
        }
        if (vlen) for (var i = 0; i < vlen; i++) {
            vchild = vchildren[i];
            child = null;
            if (keyedLen && vchild.attributes) {
                var key = vchild.key;
                if (!empty(key) && key in keyed) {
                    child = keyed[key];
                    keyed[key] = void 0;
                    keyedLen--;
                }
            }
            if (!child && min < childrenLen) for (j = min; j < childrenLen; j++) {
                c = children[j];
                if (c && isSameNodeType(c, vchild)) {
                    child = c;
                    children[j] = void 0;
                    if (j === childrenLen - 1) childrenLen--;
                    if (j === min) min++;
                    break;
                }
            }
            child = idiff(child, vchild, context, mountAll);
            if (child !== originalChildren[i]) dom.insertBefore(child, originalChildren[i] || null);
        }
        if (keyedLen) for (var i in keyed) if (keyed[i]) children[min = childrenLen++] = keyed[i];
        if (min < childrenLen) removeOrphanedChildren(children);
    }
    function removeOrphanedChildren(children, unmountOnly) {
        for (var i = children.length; i--; ) {
            var child = children[i];
            if (child) recollectNodeTree(child, unmountOnly);
        }
    }
    function recollectNodeTree(node, unmountOnly) {
        var component = node._component;
        if (component) unmountComponent(component, !unmountOnly); else {
            if (node[ATTR_KEY] && node[ATTR_KEY].ref) node[ATTR_KEY].ref(null);
            if (!unmountOnly) collectNode(node);
            if (node.childNodes && node.childNodes.length) removeOrphanedChildren(node.childNodes, unmountOnly);
        }
    }
    function diffAttributes(dom, attrs) {
        var old = dom[ATTR_KEY] || getRawNodeAttributes(dom);
        for (var _name in old) if (!(attrs && _name in attrs)) setAccessor(dom, _name, null, old[_name], isSvgMode);
        if (attrs) for (var _name2 in attrs) if (!(_name2 in old) || attrs[_name2] != old[_name2] || ('value' === _name2 || 'checked' === _name2) && attrs[_name2] != dom[_name2]) setAccessor(dom, _name2, attrs[_name2], old[_name2], isSvgMode);
    }
    function collectComponent(component) {
        var name = component.constructor.name, list = components[name];
        if (list) list.push(component); else components[name] = [ component ];
    }
    function createComponent(Ctor, props, context) {
        var inst = new Ctor(props, context), list = components[Ctor.name];
        inst.props = props;
        inst.context = context;
        if (list) for (var i = list.length; i--; ) if (list[i].constructor === Ctor) {
            inst.nextBase = list[i].nextBase;
            list.splice(i, 1);
            break;
        }
        return inst;
    }
    function triggerComponentRender(component) {
        if (!component._dirty) {
            component._dirty = !0;
            enqueueRender(component);
        }
    }
    function setComponentProps(component, props, opts, context, mountAll) {
        var b = component.base;
        if (!component._disableRendering) {
            component._disableRendering = !0;
            if (component.__ref = props.ref) delete props.ref;
            if (component.__key = props.key) delete props.key;
            if (empty(b) || mountAll) {
                if (component.componentWillMount) component.componentWillMount();
            } else if (component.componentWillReceiveProps) component.componentWillReceiveProps(props, context);
            if (context && context !== component.context) {
                if (!component.prevContext) component.prevContext = component.context;
                component.context = context;
            }
            if (!component.prevProps) component.prevProps = component.props;
            component.props = props;
            component._disableRendering = !1;
            if (0 !== opts) if (1 === opts || options.syncComponentUpdates !== !1 || !b) renderComponent(component, 1, mountAll); else triggerComponentRender(component);
            if (component.__ref) component.__ref(component);
        }
    }
    function renderComponent(component, opts, mountAll) {
        if (!component._disableRendering) {
            var skip, rendered, props = component.props, state = component.state, context = component.context, previousProps = component.prevProps || props, previousState = component.prevState || state, previousContext = component.prevContext || context, isUpdate = component.base, initialBase = isUpdate || component.nextBase, baseParent = initialBase && initialBase.parentNode, initialComponent = initialBase && initialBase._component, initialChildComponent = component._component;
            if (isUpdate) {
                component.props = previousProps;
                component.state = previousState;
                component.context = previousContext;
                if (2 !== opts && component.shouldComponentUpdate && component.shouldComponentUpdate(props, state, context) === !1) skip = !0; else if (component.componentWillUpdate) component.componentWillUpdate(props, state, context);
                component.props = props;
                component.state = state;
                component.context = context;
            }
            component.prevProps = component.prevState = component.prevContext = component.nextBase = null;
            component._dirty = !1;
            if (!skip) {
                if (component.render) rendered = component.render(props, state, context);
                if (component.getChildContext) context = extend(clone(context), component.getChildContext());
                while (isFunctionalComponent(rendered)) rendered = buildFunctionalComponent(rendered, context);
                var toUnmount, base, childComponent = rendered && rendered.nodeName;
                if (isFunction(childComponent) && childComponent.prototype.render) {
                    var inst = initialChildComponent, childProps = getNodeProps(rendered);
                    if (inst && inst.constructor === childComponent) setComponentProps(inst, childProps, 1, context); else {
                        toUnmount = inst;
                        inst = createComponent(childComponent, childProps, context);
                        inst._parentComponent = component;
                        component._component = inst;
                        setComponentProps(inst, childProps, 0, context);
                        renderComponent(inst, 1);
                    }
                    base = inst.base;
                } else {
                    var cbase = initialBase;
                    toUnmount = initialChildComponent;
                    if (toUnmount) cbase = component._component = null;
                    if (initialBase || 1 === opts) {
                        if (cbase) cbase._component = null;
                        base = diff(cbase, rendered, context, mountAll || !isUpdate, baseParent, !0, initialBase && initialBase.nextSibling);
                    }
                }
                if (initialBase && base !== initialBase) if (!toUnmount && initialComponent === component && !initialChildComponent && initialBase.parentNode) {
                    initialBase._component = null;
                    recollectNodeTree(initialBase);
                }
                if (toUnmount) unmountComponent(toUnmount, !0);
                component.base = base;
                if (base) {
                    var componentRef = component, t = component;
                    while (t = t._parentComponent) componentRef = t;
                    base._component = componentRef;
                    base._componentConstructor = componentRef.constructor;
                }
            }
            if (!isUpdate || mountAll) {
                mounts.unshift(component);
                if (!diffLevel) flushMounts();
            } else if (!skip && component.componentDidUpdate) component.componentDidUpdate(previousProps, previousState, previousContext);
            var fn, cb = component._renderCallbacks;
            if (cb) while (fn = cb.pop()) fn.call(component);
            return rendered;
        }
    }
    function buildComponentFromVNode(dom, vnode, context, mountAll) {
        var c = dom && dom._component, oldDom = dom, isDirectOwner = c && dom._componentConstructor === vnode.nodeName, isOwner = isDirectOwner, props = getNodeProps(vnode);
        while (c && !isOwner && (c = c._parentComponent)) isOwner = c.constructor === vnode.nodeName;
        if (isOwner && (!mountAll || c._component)) {
            setComponentProps(c, props, 3, context, mountAll);
            dom = c.base;
        } else {
            if (c && !isDirectOwner) {
                unmountComponent(c, !0);
                dom = oldDom = null;
            }
            c = createComponent(vnode.nodeName, props, context);
            if (dom && !c.nextBase) c.nextBase = dom;
            setComponentProps(c, props, 1, context, mountAll);
            dom = c.base;
            if (oldDom && dom !== oldDom) {
                oldDom._component = null;
                recollectNodeTree(oldDom);
            }
        }
        return dom;
    }
    function unmountComponent(component, remove) {
        var base = component.base;
        component._disableRendering = !0;
        if (component.componentWillUnmount) component.componentWillUnmount();
        component.base = null;
        var inner = component._component;
        if (inner) unmountComponent(inner, remove); else if (base) {
            if (base[ATTR_KEY] && base[ATTR_KEY].ref) base[ATTR_KEY].ref(null);
            component.nextBase = base;
            if (remove) {
                removeNode(base);
                collectComponent(component);
            }
            removeOrphanedChildren(base.childNodes, !remove);
        }
        if (component.__ref) component.__ref(null);
        if (component.componentDidUnmount) component.componentDidUnmount();
    }
    function Component(props, context) {
        this._dirty = !0;
        this._disableRendering = !1;
        this.prevState = this.prevProps = this.prevContext = this.base = this.nextBase = this._parentComponent = this._component = this.__ref = this.__key = this._linkedStates = this._renderCallbacks = null;
        this.context = context;
        this.props = props;
        this.state = this.getInitialState && this.getInitialState() || {};
    }
    function render(vnode, parent, merge) {
        return diff(merge, vnode, {}, !1, parent);
    }
    var lcCache = {};
    var toLowerCase = function(s) {
        return lcCache[s] || (lcCache[s] = s.toLowerCase());
    };
    var resolved = 'undefined' != typeof Promise && Promise.resolve();
    var defer = resolved ? function(f) {
        resolved.then(f);
    } : setTimeout;
    var options = {
        vnode: empty
    };
    var SHARED_TEMP_ARRAY = [];
    var EMPTY = {};
    var ATTR_KEY = 'undefined' != typeof Symbol ? Symbol.for('preactattr') : '__preactattr_';
    var NON_DIMENSION_PROPS = {
        boxFlex: 1,
        boxFlexGroup: 1,
        columnCount: 1,
        fillOpacity: 1,
        flex: 1,
        flexGrow: 1,
        flexPositive: 1,
        flexShrink: 1,
        flexNegative: 1,
        fontWeight: 1,
        lineClamp: 1,
        lineHeight: 1,
        opacity: 1,
        order: 1,
        orphans: 1,
        strokeOpacity: 1,
        widows: 1,
        zIndex: 1,
        zoom: 1
    };
    var items = [];
    var itemsOffline = [];
    var nodes = {};
    var mounts = [];
    var diffLevel = 0;
    var isSvgMode = !1;
    var components = {};
    extend(Component.prototype, {
        linkState: function(key, eventPath) {
            var c = this._linkedStates || (this._linkedStates = {}), cacheKey = key + '|' + eventPath;
            return c[cacheKey] || (c[cacheKey] = createLinkedState(this, key, eventPath));
        },
        setState: function(state, callback) {
            var s = this.state;
            if (!this.prevState) this.prevState = clone(s);
            extend(s, isFunction(state) ? state(s, this.props) : state);
            if (callback) (this._renderCallbacks = this._renderCallbacks || []).push(callback);
            triggerComponentRender(this);
        },
        forceUpdate: function() {
            renderComponent(this, 2);
        },
        render: function() {
            return null;
        }
    });
    exports.h = h;
    exports.cloneElement = cloneElement;
    exports.Component = Component;
    exports.render = render;
    exports.rerender = rerender;
    exports.options = options;
});
//# sourceMappingURL=preact.js.map

/***/ }),
/* 2 */
/*!*****************************************!*\
  !*** ./node_modules/process/browser.js ***!
  \*****************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

function defaultSetTimout() {
    throw new Error('setTimeout has not been defined');
}
function defaultClearTimeout () {
    throw new Error('clearTimeout has not been defined');
}
(function () {
    try {
        if (typeof setTimeout === 'function') {
            cachedSetTimeout = setTimeout;
        } else {
            cachedSetTimeout = defaultSetTimout;
        }
    } catch (e) {
        cachedSetTimeout = defaultSetTimout;
    }
    try {
        if (typeof clearTimeout === 'function') {
            cachedClearTimeout = clearTimeout;
        } else {
            cachedClearTimeout = defaultClearTimeout;
        }
    } catch (e) {
        cachedClearTimeout = defaultClearTimeout;
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    // if setTimeout wasn't available but was latter defined
    if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
        cachedSetTimeout = setTimeout;
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    // if clearTimeout wasn't available but was latter defined
    if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
        cachedClearTimeout = clearTimeout;
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;
process.prependListener = noop;
process.prependOnceListener = noop;

process.listeners = function (name) { return [] }

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };


/***/ }),
/* 3 */
/*!*********************!*\
  !*** ./src/main.js ***!
  \*********************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

var _OpenSeadragon = __webpack_require__(/*! OpenSeadragon */ 0);

var _preact = __webpack_require__(/*! preact */ 1);

var _Overlay = __webpack_require__(/*! ./views/Overlay */ 4);

var _Overlay2 = _interopRequireDefault(_Overlay);

var _Controls = __webpack_require__(/*! ./views/Controls */ 5);

var _createDispatcher = __webpack_require__(/*! ./model/createDispatcher */ 18);

var _createDispatcher2 = _interopRequireDefault(_createDispatcher);

var _generalActions = __webpack_require__(/*! ./model/generalActions */ 22);

var _generalActions2 = _interopRequireDefault(_generalActions);

var _createModel = __webpack_require__(/*! ./model/createModel */ 23);

var _createModel2 = _interopRequireDefault(_createModel);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var annotationsPrototype = {
  onOpen: function onOpen() {
    var homeBounds = this.viewer.world.getHomeBounds();
    var zoom = this.viewer.viewport.getZoom();
    this.overlay = (0, _preact.render)((0, _preact.h)(_Overlay2.default, { dispatch: this.dispatch, model: this.model }));
    this.viewer.addOverlay(this.overlay, new _OpenSeadragon.Rect(0, 0, homeBounds.width, homeBounds.height));

    var _overlay$getBoundingC = this.overlay.getBoundingClientRect(),
        width = _overlay$getBoundingC.width,
        height = _overlay$getBoundingC.height;

    this.dispatch({ type: 'INITIALIZE', zoom: zoom, width: width, height: height });
    if (!this.controls) {
      this.controls = [new _Controls.MoveControl({ dispatch: this.dispatch, model: this.model, viewer: this.viewer }), new _Controls.DrawLineControl({ dispatch: this.dispatch, model: this.model, viewer: this.viewer }), new _Controls.DrawFreeControl({ dispatch: this.dispatch, model: this.model, viewer: this.viewer })];
    }
    this.cleanAnnotations();
  },
  onClose: function onClose() {
    // TODO
  },
  getAnnotations: function getAnnotations() {
    return this.model.getAll();
  },
  setAnnotations: function setAnnotations(annotations) {
    this.dispatch({ type: 'ANNOTATIONS_RESET', annotations: annotations });
  },
  cleanAnnotations: function cleanAnnotations() {
    this.dispatch({ type: 'ANNOTATIONS_RESET' });
  },
  getMode: function getMode() {
    return this.model.mode;
  },
  setMode: function setMode(mode) {
    this.controls[0].dispatch({ type: 'MODE_UPDATE', mode: mode });
  },
  setAnnotationColor: function setAnnotationColor(color) {
    this.model.annotationcolor = color;
  },
  setAnnotationLineWidth: function setAnnotationLineWidth(linewidth) {
    this.model.annotationlinewidth = linewidth;
  },
  setAnnotationName: function setAnnotationName(name) {
    this.model.annotationname = name;
  },
  getStatus: function getStatus() {
    return { active: !!this.overlay };
  },
  EnableControls: function EnableControls(active) {
    this.model.controlsactive = !!active;
    if (this.controls) {
      for (var index = 0; index < this.controls.length; index += 1) {
        if (this.model.controlsactive) {
          this.controls[index].btn.enable();
        } else {
          this.controls[index].btn.disable();
        }
      }
    }
  },
  getLength: function getLength(annotation) {
    var linelength = 0;
    var homeBounds = this.viewer.source.dimensions;
    var ImageWidthInPercent = homeBounds.x / 100;
    var ImageHeightInPercent = homeBounds.y / 100;
    switch (annotation[0]) {
      case 'line':
        {
          var dx = (annotation[1].x2 - annotation[1].x1) * ImageWidthInPercent;
          var dy = (annotation[1].y2 - annotation[1].y1) * ImageHeightInPercent;
          linelength = Math.sqrt(dx * dx + dy * dy);
          break;
        }
      case 'path':
        {
          // remove first 'M'
          var points = annotation[1].d.substring(1).split(' L');
          if (points.length > 1) {
            for (var index = 1; index < points.length; index += 1) {
              var p1 = points[index - 1].split(' ');
              var p2 = points[index].split(' ');
              var _dx = (p1[0] - p2[0]) * ImageWidthInPercent;
              var _dy = (p1[1] - p2[1]) * ImageHeightInPercent;
              linelength += Math.sqrt(_dx * _dx + _dy * _dy);
            }
          }
          break;
        }
      default:
    }
    return linelength;
  }
};

exports.default = function (_ref) {
  var viewer = _ref.viewer;

  var model = (0, _createModel2.default)();
  var dispatch = (0, _createDispatcher2.default)(model, _generalActions2.default);
  var annotations = Object.create(annotationsPrototype);
  Object.assign(annotations, { viewer: viewer, model: model, dispatch: dispatch });
  viewer.addHandler('open', function () {
    return annotations.onOpen();
  });
  viewer.addHandler('zoom', function (_ref2) {
    var zoom = _ref2.zoom;
    return annotations.dispatch({ type: 'ZOOM_UPDATE', zoom: zoom });
  });
  if (viewer.isOpen()) {
    annotations.onOpen();
  }
  return annotations;
};

/***/ }),
/* 4 */
/*!******************************!*\
  !*** ./src/views/Overlay.js ***!
  \******************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _preact = __webpack_require__(/*! preact */ 1);

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Overlay = function (_Component) {
  _inherits(Overlay, _Component);

  function Overlay() {
    var _ref;

    _classCallCheck(this, Overlay);

    for (var _len = arguments.length, args = Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    // In Internet Explorer and Edge we can not rely on vector-effect="non-scaling-stroke"
    // to maintain constant the witdh of the SVG strokes during zoom
    var _this = _possibleConstructorReturn(this, (_ref = Overlay.__proto__ || Object.getPrototypeOf(Overlay)).call.apply(_ref, [this].concat(args)));

    _this.onMouseDown = function (e) {
      if (_this.handleMouseForAnnotation()) {
        e.stopPropagation();
        _this.props.dispatch(_extends({ type: 'PRESS' }, _this.calculateCoords(e)));
      }
    };

    _this.onMouseMove = function (e) {
      if (_this.handleMouseForAnnotation()) {
        e.stopPropagation();
        _this.props.dispatch(_extends({ type: 'MOVE' }, _this.calculateCoords(e)));
      }
    };

    _this.onMouseUp = function (e) {
      if (_this.handleMouseForAnnotation()) {
        e.stopPropagation();
        _this.props.dispatch({ type: 'RELEASE' });
      }
    };

    _this.onMouseLeave = function (e) {
      if (_this.handleMouseForAnnotation()) {
        e.stopPropagation();
        _this.props.dispatch({ type: 'LEAVE_CANVAS' });
      }
    };

    var vectorEffectNotAvailable = document.documentElement.style.vectorEffect === undefined;
    // Make sure only first 2 elements of array are used, later elements are used for name
    var standardRender = function standardRender(el) {
      return _preact.h.apply(undefined, _toConsumableArray(el.slice(0, 2)));
    };
    var renderForIeAndEdge = function renderForIeAndEdge(el, model) {
      var newEl = el.slice(0, 2);
      var baseWidth = 3;
      var percentWidth = void 0;
      var totalImageWidthInPixels = model.width * model.zoom;
      if (totalImageWidthInPixels === 0) {
        percentWidth = 0;
      } // image not yet initialized
      percentWidth = baseWidth * 100 / totalImageWidthInPixels;
      newEl[1]['stroke-width'] = percentWidth;
      return _preact.h.apply(undefined, _toConsumableArray(newEl));
    };
    _this.renderElement = vectorEffectNotAvailable ? renderForIeAndEdge : standardRender;
    return _this;
  }

  _createClass(Overlay, [{
    key: 'getInitialState',
    value: function getInitialState() {
      return this.props.model;
    }
  }, {
    key: 'componentDidMount',
    value: function componentDidMount() {
      var _this2 = this;

      this.props.model.addHandler('CHANGE_EVENT', function () {
        _this2.setState(_this2.props.model);
      });
    }
  }, {
    key: 'handleMouseForAnnotation',
    value: function handleMouseForAnnotation() {
      return this.state.mode !== 'MOVE' && this.props.model.controlsactive;
    }
  }, {
    key: 'calculateCoords',
    value: function calculateCoords(e) {
      var rect = this.base.getBoundingClientRect();
      var offsetX = e.clientX - rect.left;
      var offsetY = e.clientY - rect.top;
      var x = 100 * offsetX / rect.width;
      var y = 100 * offsetY / rect.height;
      return { x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100 };
    }
  }, {
    key: 'render',
    value: function render() {
      var _this3 = this;

      var onMouseDown = this.onMouseDown,
          onMouseLeave = this.onMouseLeave,
          onMouseMove = this.onMouseMove,
          onMouseUp = this.onMouseUp;

      return (0, _preact.h)('svg', {
        xmlns: 'http://www.w3.org/2000/svg',
        version: '1.1',
        preserveAspectRatio: 'none',
        viewBox: '0 0 100 100',
        width: '100%',
        height: '100%',
        style: {
          cursor: 'default',
          'background-color': 'rgba(0,0,0,0)' // IE 9-10 fix
        },
        onMouseDown: onMouseDown,
        onMouseLeave: onMouseLeave,
        onMouseMove: onMouseMove,
        onMouseUp: onMouseUp,
        onPointerDown: onMouseDown,
        onPointerUp: onMouseUp
      }, this.state.annotations.map(function (el) {
        return _this3.renderElement(el, _this3.state);
      }));
    }
  }]);

  return Overlay;
}(_preact.Component);

exports.default = Overlay;

/***/ }),
/* 5 */
/*!*******************************!*\
  !*** ./src/views/Controls.js ***!
  \*******************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.MoveControl = exports.DrawFreeControl = exports.DrawLineControl = exports.Control = undefined;

var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _OpenSeadragon = __webpack_require__(/*! OpenSeadragon */ 0);

var _draw_grouphover = __webpack_require__(/*! ../../img/draw_grouphover.png */ 6);

var _draw_grouphover2 = _interopRequireDefault(_draw_grouphover);

var _draw_hover = __webpack_require__(/*! ../../img/draw_hover.png */ 7);

var _draw_hover2 = _interopRequireDefault(_draw_hover);

var _draw_pressed = __webpack_require__(/*! ../../img/draw_pressed.png */ 8);

var _draw_pressed2 = _interopRequireDefault(_draw_pressed);

var _draw_rest = __webpack_require__(/*! ../../img/draw_rest.png */ 9);

var _draw_rest2 = _interopRequireDefault(_draw_rest);

var _path_grouphover = __webpack_require__(/*! ../../img/path_grouphover.png */ 10);

var _path_grouphover2 = _interopRequireDefault(_path_grouphover);

var _path_hover = __webpack_require__(/*! ../../img/path_hover.png */ 11);

var _path_hover2 = _interopRequireDefault(_path_hover);

var _path_pressed = __webpack_require__(/*! ../../img/path_pressed.png */ 12);

var _path_pressed2 = _interopRequireDefault(_path_pressed);

var _path_rest = __webpack_require__(/*! ../../img/path_rest.png */ 13);

var _path_rest2 = _interopRequireDefault(_path_rest);

var _move_grouphover = __webpack_require__(/*! ../../img/move_grouphover.png */ 14);

var _move_grouphover2 = _interopRequireDefault(_move_grouphover);

var _move_hover = __webpack_require__(/*! ../../img/move_hover.png */ 15);

var _move_hover2 = _interopRequireDefault(_move_hover);

var _move_pressed = __webpack_require__(/*! ../../img/move_pressed.png */ 16);

var _move_pressed2 = _interopRequireDefault(_move_pressed);

var _move_rest = __webpack_require__(/*! ../../img/move_rest.png */ 17);

var _move_rest2 = _interopRequireDefault(_move_rest);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Control = exports.Control = function () {
  function Control(options) {
    var _this = this;

    _classCallCheck(this, Control);

    this.dispatch = options.dispatch;
    this.model = options.model;
    this.viewer = options.viewer;
    this.mode = options.Tooltip.toUpperCase();
    this.btn = new _OpenSeadragon.Button((0, _OpenSeadragon.extend)({
      onClick: function onClick(e) {
        _this.onClick(e);
      }
    }, options));
    this.viewer.addControl(this.btn.element, {
      anchor: _OpenSeadragon.ControlAnchor.TOP_LEFT
    });
    if (this.model.mode === this.mode) {
      this.activate();
    }
    if (this.model.controlsactive) {
      this.btn.enable();
    } else {
      this.btn.disable();
    }
    this.model.addHandler('CHANGE_EVENT', function () {
      if (_this.model.mode === _this.mode) {
        _this.activate();
      } else {
        _this.deactivate();
      }
    });
  }

  _createClass(Control, [{
    key: 'activate',
    value: function activate() {
      this.btn.imgDown.style.visibility = 'visible';
    }
  }, {
    key: 'deactivate',
    value: function deactivate() {
      this.btn.imgDown.style.visibility = 'hidden';
    }
  }, {
    key: 'onClick',
    value: function onClick(_ref) {
      var eventSource = _ref.eventSource;

      if (eventSource.Tooltip) {
        var mode = eventSource.Tooltip.toUpperCase();
        this.dispatch({ type: 'MODE_UPDATE', mode: mode });
      }
    }
  }]);

  return Control;
}();

var DrawLineControl = exports.DrawLineControl = function (_Control) {
  _inherits(DrawLineControl, _Control);

  function DrawLineControl(options) {
    _classCallCheck(this, DrawLineControl);

    return _possibleConstructorReturn(this, (DrawLineControl.__proto__ || Object.getPrototypeOf(DrawLineControl)).call(this, _extends({
      Tooltip: 'LineDraw',
      srcRest: _draw_rest2.default,
      srcGroup: _draw_grouphover2.default,
      srcHover: _draw_hover2.default,
      srcDown: _draw_pressed2.default
    }, options)));
  }

  return DrawLineControl;
}(Control);

var DrawFreeControl = exports.DrawFreeControl = function (_Control2) {
  _inherits(DrawFreeControl, _Control2);

  function DrawFreeControl(options) {
    _classCallCheck(this, DrawFreeControl);

    return _possibleConstructorReturn(this, (DrawFreeControl.__proto__ || Object.getPrototypeOf(DrawFreeControl)).call(this, _extends({
      Tooltip: 'FreeDraw',
      srcRest: _path_rest2.default,
      srcGroup: _path_grouphover2.default,
      srcHover: _path_hover2.default,
      srcDown: _path_pressed2.default
    }, options)));
  }

  return DrawFreeControl;
}(Control);

var MoveControl = exports.MoveControl = function (_Control3) {
  _inherits(MoveControl, _Control3);

  function MoveControl(options) {
    _classCallCheck(this, MoveControl);

    return _possibleConstructorReturn(this, (MoveControl.__proto__ || Object.getPrototypeOf(MoveControl)).call(this, _extends({
      Tooltip: 'Move',
      srcRest: _move_rest2.default,
      srcGroup: _move_grouphover2.default,
      srcHover: _move_hover2.default,
      srcDown: _move_pressed2.default
    }, options)));
  }

  return MoveControl;
}(Control);

/***/ }),
/* 6 */
/*!*********************************!*\
  !*** ./img/draw_grouphover.png ***!
  \*********************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDozRkRERTMzMkVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDozRkRERTMzMUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 7 */
/*!****************************!*\
  !*** ./img/draw_hover.png ***!
  \****************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDoyMUZEMjkyNUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDoyMUZEMjkyNEVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 8 */
/*!******************************!*\
  !*** ./img/draw_pressed.png ***!
  \******************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDozRkRERTMzQUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDozRkRERTMzOUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 9 */
/*!***************************!*\
  !*** ./img/draw_rest.png ***!
  \***************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDowRUI0OEJGRUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDowRUI0OEJGREVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 10 */
/*!*********************************!*\
  !*** ./img/path_grouphover.png ***!
  \*********************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,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"

/***/ }),
/* 11 */
/*!****************************!*\
  !*** ./img/path_hover.png ***!
  \****************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,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"

/***/ }),
/* 12 */
/*!******************************!*\
  !*** ./img/path_pressed.png ***!
  \******************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,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"

/***/ }),
/* 13 */
/*!***************************!*\
  !*** ./img/path_rest.png ***!
  \***************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,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"

/***/ }),
/* 14 */
/*!*********************************!*\
  !*** ./img/move_grouphover.png ***!
  \*********************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDoyMUZEMjkyOUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDoyMUZEMjkyOEVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 15 */
/*!****************************!*\
  !*** ./img/move_hover.png ***!
  \****************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDoyMUZEMjkyMUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDoyMUZEMjkyMEVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 16 */
/*!******************************!*\
  !*** ./img/move_pressed.png ***!
  \******************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDozRkRERTMzNkVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDozRkRERTMzNUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 17 */
/*!***************************!*\
  !*** ./img/move_rest.png ***!
  \***************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports) {

module.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAiCAYAAADVhWD8AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDowMTgwMTE3NDA3MjA2ODExODIyQUEwM0FBMUQ5NUNBNSIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDowRUI0OEJGQUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDowRUI0OEJGOUVGRDYxMUUzQTExNUI3MjM4QzI2QjRFRiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M2IChNYWNpbnRvc2gpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MjhBOTkxNUEwRTIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6MDE4MDExNzQwNzIwNjgxMTgyMkFBMDNBQTFEOTVDQTUiLz4gPC9yZGY6RGVzY3Jp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"

/***/ }),
/* 18 */
/*!***************************************!*\
  !*** ./src/model/createDispatcher.js ***!
  \***************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

var _flux = __webpack_require__(/*! flux */ 19);

var createDispatcher = function createDispatcher(model) {
  for (var _len = arguments.length, reactions = Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    reactions[_key - 1] = arguments[_key];
  }

  var dispatcher = new _flux.Dispatcher();
  reactions.forEach(function (reaction) {
    dispatcher.register(reaction(model));
  });
  return function (action) {
    return dispatcher.dispatch(action);
  };
};

exports.default = createDispatcher;

/***/ }),
/* 19 */
/*!************************************!*\
  !*** ./node_modules/flux/index.js ***!
  \************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

/**
 * Copyright (c) 2014-2015, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

module.exports.Dispatcher = __webpack_require__(/*! ./lib/Dispatcher */ 20);


/***/ }),
/* 20 */
/*!*********************************************!*\
  !*** ./node_modules/flux/lib/Dispatcher.js ***!
  \*********************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
/* WEBPACK VAR INJECTION */(function(process) {/**
 * Copyright (c) 2014-2015, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 *
 * @providesModule Dispatcher
 * 
 * @preventMunge
 */



exports.__esModule = true;

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError('Cannot call a class as a function'); } }

var invariant = __webpack_require__(/*! fbjs/lib/invariant */ 21);

var _prefix = 'ID_';

/**
 * Dispatcher is used to broadcast payloads to registered callbacks. This is
 * different from generic pub-sub systems in two ways:
 *
 *   1) Callbacks are not subscribed to particular events. Every payload is
 *      dispatched to every registered callback.
 *   2) Callbacks can be deferred in whole or part until other callbacks have
 *      been executed.
 *
 * For example, consider this hypothetical flight destination form, which
 * selects a default city when a country is selected:
 *
 *   var flightDispatcher = new Dispatcher();
 *
 *   // Keeps track of which country is selected
 *   var CountryStore = {country: null};
 *
 *   // Keeps track of which city is selected
 *   var CityStore = {city: null};
 *
 *   // Keeps track of the base flight price of the selected city
 *   var FlightPriceStore = {price: null}
 *
 * When a user changes the selected city, we dispatch the payload:
 *
 *   flightDispatcher.dispatch({
 *     actionType: 'city-update',
 *     selectedCity: 'paris'
 *   });
 *
 * This payload is digested by `CityStore`:
 *
 *   flightDispatcher.register(function(payload) {
 *     if (payload.actionType === 'city-update') {
 *       CityStore.city = payload.selectedCity;
 *     }
 *   });
 *
 * When the user selects a country, we dispatch the payload:
 *
 *   flightDispatcher.dispatch({
 *     actionType: 'country-update',
 *     selectedCountry: 'australia'
 *   });
 *
 * This payload is digested by both stores:
 *
 *   CountryStore.dispatchToken = flightDispatcher.register(function(payload) {
 *     if (payload.actionType === 'country-update') {
 *       CountryStore.country = payload.selectedCountry;
 *     }
 *   });
 *
 * When the callback to update `CountryStore` is registered, we save a reference
 * to the returned token. Using this token with `waitFor()`, we can guarantee
 * that `CountryStore` is updated before the callback that updates `CityStore`
 * needs to query its data.
 *
 *   CityStore.dispatchToken = flightDispatcher.register(function(payload) {
 *     if (payload.actionType === 'country-update') {
 *       // `CountryStore.country` may not be updated.
 *       flightDispatcher.waitFor([CountryStore.dispatchToken]);
 *       // `CountryStore.country` is now guaranteed to be updated.
 *
 *       // Select the default city for the new country
 *       CityStore.city = getDefaultCityForCountry(CountryStore.country);
 *     }
 *   });
 *
 * The usage of `waitFor()` can be chained, for example:
 *
 *   FlightPriceStore.dispatchToken =
 *     flightDispatcher.register(function(payload) {
 *       switch (payload.actionType) {
 *         case 'country-update':
 *         case 'city-update':
 *           flightDispatcher.waitFor([CityStore.dispatchToken]);
 *           FlightPriceStore.price =
 *             getFlightPriceStore(CountryStore.country, CityStore.city);
 *           break;
 *     }
 *   });
 *
 * The `country-update` payload will be guaranteed to invoke the stores'
 * registered callbacks in order: `CountryStore`, `CityStore`, then
 * `FlightPriceStore`.
 */

var Dispatcher = (function () {
  function Dispatcher() {
    _classCallCheck(this, Dispatcher);

    this._callbacks = {};
    this._isDispatching = false;
    this._isHandled = {};
    this._isPending = {};
    this._lastID = 1;
  }

  /**
   * Registers a callback to be invoked with every dispatched payload. Returns
   * a token that can be used with `waitFor()`.
   */

  Dispatcher.prototype.register = function register(callback) {
    var id = _prefix + this._lastID++;
    this._callbacks[id] = callback;
    return id;
  };

  /**
   * Removes a callback based on its token.
   */

  Dispatcher.prototype.unregister = function unregister(id) {
    !this._callbacks[id] ? process.env.NODE_ENV !== 'production' ? invariant(false, 'Dispatcher.unregister(...): `%s` does not map to a registered callback.', id) : invariant(false) : undefined;
    delete this._callbacks[id];
  };

  /**
   * Waits for the callbacks specified to be invoked before continuing execution
   * of the current callback. This method should only be used by a callback in
   * response to a dispatched payload.
   */

  Dispatcher.prototype.waitFor = function waitFor(ids) {
    !this._isDispatching ? process.env.NODE_ENV !== 'production' ? invariant(false, 'Dispatcher.waitFor(...): Must be invoked while dispatching.') : invariant(false) : undefined;
    for (var ii = 0; ii < ids.length; ii++) {
      var id = ids[ii];
      if (this._isPending[id]) {
        !this._isHandled[id] ? process.env.NODE_ENV !== 'production' ? invariant(false, 'Dispatcher.waitFor(...): Circular dependency detected while ' + 'waiting for `%s`.', id) : invariant(false) : undefined;
        continue;
      }
      !this._callbacks[id] ? process.env.NODE_ENV !== 'production' ? invariant(false, 'Dispatcher.waitFor(...): `%s` does not map to a registered callback.', id) : invariant(false) : undefined;
      this._invokeCallback(id);
    }
  };

  /**
   * Dispatches a payload to all registered callbacks.
   */

  Dispatcher.prototype.dispatch = function dispatch(payload) {
    !!this._isDispatching ? process.env.NODE_ENV !== 'production' ? invariant(false, 'Dispatch.dispatch(...): Cannot dispatch in the middle of a dispatch.') : invariant(false) : undefined;
    this._startDispatching(payload);
    try {
      for (var id in this._callbacks) {
        if (this._isPending[id]) {
          continue;
        }
        this._invokeCallback(id);
      }
    } finally {
      this._stopDispatching();
    }
  };

  /**
   * Is this Dispatcher currently dispatching.
   */

  Dispatcher.prototype.isDispatching = function isDispatching() {
    return this._isDispatching;
  };

  /**
   * Call the callback stored with the given id. Also do some internal
   * bookkeeping.
   *
   * @internal
   */

  Dispatcher.prototype._invokeCallback = function _invokeCallback(id) {
    this._isPending[id] = true;
    this._callbacks[id](this._pendingPayload);
    this._isHandled[id] = true;
  };

  /**
   * Set up bookkeeping needed when dispatching.
   *
   * @internal
   */

  Dispatcher.prototype._startDispatching = function _startDispatching(payload) {
    for (var id in this._callbacks) {
      this._isPending[id] = false;
      this._isHandled[id] = false;
    }
    this._pendingPayload = payload;
    this._isDispatching = true;
  };

  /**
   * Clear bookkeeping used for dispatching.
   *
   * @internal
   */

  Dispatcher.prototype._stopDispatching = function _stopDispatching() {
    delete this._pendingPayload;
    this._isDispatching = false;
  };

  return Dispatcher;
})();

module.exports = Dispatcher;
/* WEBPACK VAR INJECTION */}.call(exports, __webpack_require__(/*! ./../../process/browser.js */ 2)))

/***/ }),
/* 21 */
/*!********************************************!*\
  !*** ./node_modules/fbjs/lib/invariant.js ***!
  \********************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
/* WEBPACK VAR INJECTION */(function(process) {/**
 * Copyright 2013-2015, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 *
 * @providesModule invariant
 */



/**
 * Use invariant() to assert state which your program assumes to be true.
 *
 * Provide sprintf-style format (only %s is supported) and arguments
 * to provide information about what broke and what you were
 * expecting.
 *
 * The invariant message will be stripped in production, but the invariant
 * will remain to ensure logic does not differ in production.
 */

var invariant = function (condition, format, a, b, c, d, e, f) {
  if (process.env.NODE_ENV !== 'production') {
    if (format === undefined) {
      throw new Error('invariant requires an error message argument');
    }
  }

  if (!condition) {
    var error;
    if (format === undefined) {
      error = new Error('Minified exception occurred; use the non-minified dev environment ' + 'for the full error message and additional helpful warnings.');
    } else {
      var args = [a, b, c, d, e, f];
      var argIndex = 0;
      error = new Error('Invariant Violation: ' + format.replace(/%s/g, function () {
        return args[argIndex++];
      }));
    }

    error.framesToPop = 1; // we don't care about invariant's own frame
    throw error;
  }
};

module.exports = invariant;
/* WEBPACK VAR INJECTION */}.call(exports, __webpack_require__(/*! ./../../process/browser.js */ 2)))

/***/ }),
/* 22 */
/*!*************************************!*\
  !*** ./src/model/generalActions.js ***!
  \*************************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});
var reactToGeneralAction = function reactToGeneralAction(model) {
  return function (action) {
    switch (action.type) {
      case 'MODE_UPDATE':
        model.activityInProgress = false;
        if (model.mode !== action.mode) {
          model.mode = action.mode;
        }
        break;

      case 'ACTIVITY_UPDATE':
        model.activityInProgress = action.inProgress;
        break;

      case 'PRESS':
        if ((model.mode === 'LINEDRAW' || model.mode === 'FREEDRAW') && model.controlsactive) {
          // Remove existing annotation if it has same non-empty name
          if (model.annotationname !== '') {
            var i = model.getAnnotationsIdxByName(model.annotationname);
            if (i >= 0) {
              model.annotations.splice(i, 1);
            }
          }
          model.activityInProgress = true;
          switch (model.mode) {
            case 'LINEDRAW':
              model.annotations.push(['line', {
                x1: '' + action.x,
                y1: '' + action.y,
                x2: '' + action.x,
                y2: '' + action.y,
                stroke: '' + model.annotationcolor,
                'stroke-width': '' + model.annotationlinewidth,
                'vector-effect': 'non-scaling-stroke'
              }, '' + model.annotationname]);
              break;
            case 'FREEDRAW':
              model.annotations.push(['path', {
                fill: 'none',
                d: 'M' + action.x + ' ' + action.y,
                stroke: '' + model.annotationcolor,
                'stroke-width': '' + model.annotationlinewidth,
                'stroke-linejoin': 'round',
                'stroke-linecap': 'round',
                'vector-effect': 'non-scaling-stroke'
              }, '' + model.annotationname]);
              break;
            default:
              break;
          }
        }
        break;

      case 'LEAVE_CANVAS':
      case 'RELEASE':
        if ((model.mode === 'LINEDRAW' || model.mode === 'FREEDRAW') && model.activityInProgress === true) {
          model.raiseEvent('ANNOTATIONRELEASE_EVENT', model.annotations[model.annotations.length - 1]);
          model.activityInProgress = false;
        }
        break;

      case 'MOVE':
        if ((model.mode === 'LINEDRAW' || model.mode === 'FREEDRAW') && model.activityInProgress === true) {
          var lastAnnotation = model.annotations[model.annotations.length - 1];
          if (lastAnnotation && lastAnnotation[0] === 'path') {
            lastAnnotation[1].d += ' L' + action.x + ' ' + action.y;
          } else if (lastAnnotation && lastAnnotation[0] === 'line') {
            lastAnnotation[1].x2 = '' + action.x;
            lastAnnotation[1].y2 = '' + action.y;
          }
        }
        break;

      case 'ANNOTATIONS_RESET':
        model.activityInProgress = false;
        model.annotations = action.annotations || [];
        break;

      case 'ZOOM_UPDATE':
        model.zoom = action.zoom;
        break;

      case 'INITIALIZE':
        model.zoom = action.zoom;
        model.width = action.width;
        model.height = action.height;
        break;

      default:
        break;
    }

    model.raiseEvent('CHANGE_EVENT');
  };
};

exports.default = reactToGeneralAction;

/***/ }),
/* 23 */
/*!**********************************!*\
  !*** ./src/model/createModel.js ***!
  \**********************************/
/*! dynamic exports provided */
/*! all exports used */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
  value: true
});

var _OpenSeadragon = __webpack_require__(/*! OpenSeadragon */ 0);

exports.default = function () {
  return Object.assign(Object.create(_OpenSeadragon.EventSource.prototype), {
    events: {},
    mode: 'MOVE',
    zoom: 1,
    width: 0,
    height: 0,
    activityInProgress: false,
    annotations: [],
    annotationcolor: 'green',
    annotationlinewidth: 3,
    annotationname: '',
    controlsactive: true,

    getAnnotationsIdxByName: function getAnnotationsIdxByName(annotationname) {
      var i = 0;
      while (i < this.annotations.length) {
        if (annotationname === this.annotations[i][2]) {
          return i;
        }
        i += 1;
      }
      return -1;
    },
    getAnnotationByName: function getAnnotationByName(annotationname) {
      var i = this.getAnnotationsIdxByName(annotationname);
      if (i >= 0) {
        return this.annotations[i];
      }
      return null;
    }
  });
};

/***/ })
/******/ ])["default"];
//# sourceMappingURL=openseadragon-annotations.js.map