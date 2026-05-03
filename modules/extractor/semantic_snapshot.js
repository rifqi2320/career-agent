() => {
  const INTERACTIVE_ROLES = new Set([
    'button',
    'checkbox',
    'combobox',
    'link',
    'listbox',
    'menuitem',
    'menuitemcheckbox',
    'menuitemradio',
    'option',
    'radio',
    'searchbox',
    'slider',
    'spinbutton',
    'switch',
    'tab',
    'textbox',
  ]);
  const STRUCTURAL_ROLES = new Set([
    'generic',
    'group',
    'none',
    'presentation',
    'region',
    'section',
  ]);
  const CONTAINER_ROLES = new Set([
    'article',
    'complementary',
    'contentinfo',
    'dialog',
    'feed',
    'figure',
    'form',
    'group',
    'heading',
    'list',
    'listitem',
    'log',
    'main',
    'navigation',
    'region',
    'row',
    'rowheader',
    'table',
    'tablist',
    'term',
    'toolbar',
  ]);

  const normalizeText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const quote = (value) => JSON.stringify(String(value));

  function isSkippableElement(element) {
    return ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(element.tagName);
  }

  function getControlledBy(element) {
    if (!element.id) {
      return null;
    }
    for (const controller of Array.from(document.querySelectorAll('[aria-controls]'))) {
      const controlledIds = String(controller.getAttribute('aria-controls') || '')
        .split(/\s+/)
        .filter(Boolean);
      if (controlledIds.includes(element.id)) {
        return controller;
      }
    }
    return null;
  }

  function isInPageContent(element) {
    return Boolean(element.closest('main, article, [role="main"], [role="article"]'));
  }

  function isCollapsedDisclosurePanel(element) {
    const controller = getControlledBy(element);
    if (!controller || !isInPageContent(element)) {
      return false;
    }
    const controllerTag = controller.tagName.toLowerCase();
    const controllerRole = getRole(controller);
    return (
      controller.getAttribute('aria-expanded') === 'false' &&
      (controllerTag === 'button' ||
        controllerTag === 'summary' ||
        INTERACTIVE_ROLES.has(controllerRole))
    );
  }

  function isElementHidden(element, includeHiddenDisclosureContent = false) {
    if (!element || !(element instanceof Element)) {
      return true;
    }
    if (includeHiddenDisclosureContent) {
      return false;
    }
    if (isCollapsedDisclosurePanel(element)) {
      return false;
    }
    if (element.hidden || element.getAttribute('aria-hidden') === 'true') {
      return true;
    }
    const style = window.getComputedStyle(element);
    if (!style) {
      return false;
    }
    return (
      style.display === 'none' ||
      style.visibility === 'hidden' ||
      style.visibility === 'collapse'
    );
  }

  function getImplicitRole(element) {
    const tag = element.tagName.toLowerCase();
    if (tag === 'a' && element.hasAttribute('href')) return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return element.multiple ? 'listbox' : 'combobox';
    if (tag === 'option') return 'option';
    if (tag === 'img') return 'img';
    if (tag === 'nav') return 'navigation';
    if (tag === 'main') return 'main';
    if (tag === 'article') return 'article';
    if (tag === 'section') return 'region';
    if (tag === 'form') return 'form';
    if (tag === 'dialog') return 'dialog';
    if (tag === 'ul' || tag === 'ol') return 'none';
    if (tag === 'li') return 'listitem';
    if (tag === 'table') return 'table';
    if (tag === 'tr') return 'row';
    if (tag === 'th') return 'columnheader';
    if (tag === 'td') return 'cell';
    if (tag === 'summary') return 'button';
    if (/^h[1-6]$/.test(tag)) return 'heading';
    if (tag === 'input') {
      const type = (element.getAttribute('type') || 'text').toLowerCase();
      if (['button', 'submit', 'reset', 'image'].includes(type)) return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'range') return 'slider';
      if (type === 'number') return 'spinbutton';
      if (type === 'search') return 'searchbox';
      return 'textbox';
    }
    return 'generic';
  }

  function getRole(element) {
    return element.getAttribute('role') || getImplicitRole(element);
  }

  function getTextFromIds(value) {
    return normalizeText(
      String(value || '')
        .split(/\s+/)
        .map((id) => {
          const target = document.getElementById(id);
          return target ? target.innerText || target.textContent || '' : '';
        })
        .join(' ')
    );
  }

  function getAccessibleName(element) {
    const ariaLabel = normalizeText(element.getAttribute('aria-label'));
    if (ariaLabel) return ariaLabel;

    const labelledBy = normalizeText(element.getAttribute('aria-labelledby'));
    if (labelledBy) {
      const resolved = getTextFromIds(labelledBy);
      if (resolved) return resolved;
    }

    if (element instanceof HTMLImageElement) {
      const alt = normalizeText(element.alt);
      if (alt) return alt;
    }

    if (element instanceof HTMLInputElement) {
      if (element.labels && element.labels.length > 0) {
        const labelText = normalizeText(
          Array.from(element.labels)
            .map((label) => label.innerText || label.textContent || '')
            .join(' ')
        );
        if (labelText) return labelText;
      }
      if (['button', 'submit', 'reset'].includes((element.type || '').toLowerCase())) {
        const buttonValue = normalizeText(element.value);
        if (buttonValue) return buttonValue;
      }
    }

    const title = normalizeText(element.getAttribute('title'));
    if (title) return title;

    if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
      const placeholder = normalizeText(element.getAttribute('placeholder'));
      if (placeholder) return placeholder;
    }

    const tag = element.tagName.toLowerCase();
    if (['button', 'a', 'option', 'summary', 'label'].includes(tag) || /^h[1-6]$/.test(tag)) {
      const text = normalizeText(element.innerText || element.textContent || '');
      if (text) return text;
    }

    return '';
  }

  function getElementState(element, role) {
    const state = {};
    const ariaChecked = element.getAttribute('aria-checked');
    if (ariaChecked === 'mixed') state.checked = 'mixed';
    else if (ariaChecked === 'true') state.checked = true;
    else if (ariaChecked === 'false') state.checked = false;
    else if (element instanceof HTMLInputElement && ['checkbox', 'radio'].includes(element.type)) {
      state.checked = element.checked;
    }

    if (element.hasAttribute('disabled') || element.getAttribute('aria-disabled') === 'true') {
      state.disabled = true;
    }

    const expanded = element.getAttribute('aria-expanded');
    if (expanded === 'true') state.expanded = true;
    else if (expanded === 'false') state.expanded = false;

    const pressed = element.getAttribute('aria-pressed');
    if (pressed === 'mixed') state.pressed = 'mixed';
    else if (pressed === 'true') state.pressed = true;
    else if (pressed === 'false') state.pressed = false;

    const selected = element.getAttribute('aria-selected');
    if (selected === 'true') state.selected = true;
    else if (selected === 'false') state.selected = false;
    else if (element instanceof HTMLOptionElement) state.selected = element.selected;

    if (role === 'heading') {
      const match = element.tagName.match(/^H([1-6])$/);
      const ariaLevel = element.getAttribute('aria-level');
      if (match) state.level = Number.parseInt(match[1], 10);
      else if (ariaLevel) state.level = Number.parseInt(ariaLevel, 10);
    }

    return state;
  }

  function getElementProps(element, role) {
    const props = {};
    if (role === 'link' && element.href) {
      props.url = element.href;
    }
    if (element instanceof HTMLInputElement) {
      props.inputType = element.type || 'text';
      if (element.type !== 'password' && element.value) {
        props.value = element.value;
      }
    } else if (element instanceof HTMLTextAreaElement) {
      if (element.value) props.value = element.value;
    } else if (element instanceof HTMLSelectElement) {
      props.value = element.value;
    }
    return props;
  }

  function getDirectTextChildren(element) {
    const parts = [];
    for (const child of Array.from(element.childNodes)) {
      if (child.nodeType !== Node.TEXT_NODE) continue;
      const text = normalizeText(child.textContent || '');
      if (text) parts.push(text);
    }
    return parts;
  }

  function getChildNodes(element) {
    if (element.shadowRoot) {
      return Array.from(element.shadowRoot.childNodes);
    }
    return Array.from(element.childNodes);
  }

  function shouldIncludeElement(element, role, name, props, childItems) {
    if (INTERACTIVE_ROLES.has(role) || CONTAINER_ROLES.has(role)) return true;
    if (name) return true;
    if (Object.keys(props).length > 0) return true;
    if (element.isContentEditable) return true;
    const tag = element.tagName.toLowerCase();
    if (['p', 'label', 'details', 'summary', 'img'].includes(tag)) return true;
    return childItems.length > 1;
  }

  function hasMeaningfulState(state) {
    return ['checked', 'disabled', 'expanded', 'level', 'pressed', 'selected'].some(
      (key) => state[key] !== undefined && state[key] !== false
    );
  }

  function shouldAssignRef(role, name, props, state, element) {
    if (INTERACTIVE_ROLES.has(role)) return true;
    if (
      element instanceof HTMLInputElement ||
      element instanceof HTMLTextAreaElement ||
      element instanceof HTMLSelectElement
    ) {
      return true;
    }
    if (element.isContentEditable) return true;
    if (role === 'img' || role === 'dialog' || role === 'heading') return true;
    if (name) return true;
    if (Object.keys(props).length > 0) return true;
    return hasMeaningfulState(state);
  }

  function shouldFlattenNode(node) {
    if (!STRUCTURAL_ROLES.has(node.role)) return false;
    if (node.name) return false;
    if (node.ref) return false;
    if (Object.keys(node.props || {}).length > 0) return false;
    if (hasMeaningfulState(node)) return false;
    return node.children.length > 0;
  }

  function simplifyNode(node) {
    if (typeof node === 'string') {
      return node;
    }

    const simplifiedChildren = [];
    for (const child of node.children || []) {
      const simplifiedChild = simplifyNode(child);
      if (simplifiedChild == null) continue;
      if (Array.isArray(simplifiedChild)) simplifiedChildren.push(...simplifiedChild);
      else simplifiedChildren.push(simplifiedChild);
    }
    node.children = simplifiedChildren;

    if (shouldFlattenNode(node)) {
      return node.children;
    }

    return node;
  }

  function appendStateBits(label, node) {
    let output = label;
    if (node.checked === 'mixed') output += ' [checked=mixed]';
    else if (node.checked === true) output += ' [checked]';
    if (node.disabled) output += ' [disabled]';
    if (node.expanded === true) output += ' [expanded]';
    if (node.level) output += ` [level=${node.level}]`;
    if (node.pressed === 'mixed') output += ' [pressed=mixed]';
    else if (node.pressed === true) output += ' [pressed]';
    if (node.selected === true) output += ' [selected]';
    if (node.ref) output += ` [ref=s1e${node.ref}]`;
    return output;
  }

  function serializeYaml(rootNode) {
    const lines = [];

    function walk(node, indent) {
      if (typeof node === 'string') {
        lines.push(`${indent}- text: ${quote(node)}`);
        return;
      }

      let label = node.role;
      if (node.name) label += ` ${quote(node.name)}`;
      lines.push(`${indent}- ${appendStateBits(label, node)}`);

      for (const [key, value] of Object.entries(node.props || {})) {
        lines.push(`${indent}  /${key}: ${quote(value)}`);
      }

      for (const child of node.children || []) {
        walk(child, `${indent}  `);
      }
    }

    walk(rootNode, '');
    return lines.join('\n');
  }

  function buildItems(node, context) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = normalizeText(node.textContent || '');
      return text ? [text] : [];
    }

    if (!(node instanceof Element) || isSkippableElement(node)) {
      return [];
    }

    const isHiddenDisclosurePanel = isCollapsedDisclosurePanel(node);
    const includeHiddenDisclosureContent =
      context.hiddenDisclosureDepth > 0 || isHiddenDisclosurePanel;
    if (isElementHidden(node, includeHiddenDisclosureContent)) {
      return [];
    }

    const role = getRole(node);
    const name = getAccessibleName(node);
    const props = getElementProps(node, role);
    const children = [];
    const previousHiddenDisclosureDepth = context.hiddenDisclosureDepth;
    if (isHiddenDisclosurePanel) {
      context.hiddenDisclosureDepth += 1;
    }

    for (const text of getDirectTextChildren(node)) {
      children.push(text);
    }
    for (const child of getChildNodes(node)) {
      if (child.nodeType === Node.TEXT_NODE) continue;
      children.push(...buildItems(child, context));
    }
    context.hiddenDisclosureDepth = previousHiddenDisclosureDepth;

    const includeSelf = shouldIncludeElement(node, role, name, props, children);
    if (!includeSelf) {
      return children;
    }

    const state = getElementState(node, role);
    const refId = shouldAssignRef(role, name, props, state, node) ? ++context.nextId : null;
    const item = {
      role,
      name,
      props,
      children,
      ...state,
    };
    if (refId != null) {
      item.ref = refId;
    }
    return [item];
  }

  const context = { nextId: 0, hiddenDisclosureDepth: 0 };
  const children = buildItems(document.documentElement, context);
  const root = {
    role: 'document',
    name: normalizeText(document.title || ''),
    props: { url: location.href },
    children,
  };
  root.children = root.children
    .map((child) => simplifyNode(child))
    .flat()
    .filter(Boolean);

  return {
    url: location.href,
    title: document.title,
    yaml: serializeYaml(root),
  };
}
