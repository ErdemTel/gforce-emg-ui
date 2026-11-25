"""Six multiplier controls and explicit, acknowledged bridge configuration."""

import json
import threading

from PyQt5 import QtCore, QtWidgets

from ..live import gain_table


class GainTablePanel(QtWidgets.QGroupBox):
    completed = QtCore.pyqtSignal(object, int, str)
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, store, target, parent=None, request=gain_table.request_table):
        super().__init__('Stiffness gains', parent)
        self.store, self.target, self.request = store, target, request
        self._snapshot = None
        self._busy = self._sending = self._closed = False
        self._token = 0
        self._cancel = threading.Event()
        self._thread = None
        self.kp, self.kd = [], []
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Each multiplier scales the hand’s 22 joint-specific baseline '
                               'gains. Applying changes the bridge’s table, not the hand’s '
                               'current gains. Disarm the bridge before applying.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Hand'))
        self.side = QtWidgets.QComboBox()
        self.side.addItems(['right', 'left'])
        self.side.currentTextChanged.connect(self.target_changed)
        row.addWidget(self.side)
        row.addStretch()
        layout.addLayout(row)
        grid = QtWidgets.QGridLayout()
        for col, label in enumerate(('Stiffness', 'Kp multiplier', 'Kd multiplier')):
            grid.addWidget(QtWidgets.QLabel(label), 0, col)
        for row, level in enumerate(gain_table.LEVELS, 1):
            grid.addWidget(QtWidgets.QLabel(f'{level:g}'), row, 0)
            for col, widgets, low in ((1, self.kp, .2), (2, self.kd, .1)):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(low, 1.)
                spin.setDecimals(4)
                spin.setSingleStep(.05)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(self._edited)
                widgets.append(spin)
                grid.addWidget(spin, row, col)
        layout.addLayout(grid)
        row = QtWidgets.QGridLayout()
        self.read_button = QtWidgets.QPushButton('Read bridge')
        self.apply_button = QtWidgets.QPushButton('Apply table')
        self.defaults_button = QtWidgets.QPushButton('Defaults')
        self.use_active_button = QtWidgets.QPushButton('Use bridge values')
        self.read_button.clicked.connect(lambda: self._start('read'))
        self.apply_button.clicked.connect(lambda: self._start('apply'))
        self.defaults_button.clicked.connect(
            lambda: self._set_draft(gain_table.DEFAULT_KP, gain_table.DEFAULT_KD))
        self.use_active_button.clicked.connect(self._use_active)
        for index, button in enumerate((self.read_button, self.apply_button,
                                        self.use_active_button, self.defaults_button)):
            row.addWidget(button, index // 2, index % 2)
        layout.addLayout(row)
        self.active_label = self._label('Bridge table not read. These values are a local draft.')
        self.draft_label = self._label('')
        self.status_label = self._label('Read bridge to confirm the current table. '
                                        'A bridge restart reloads its launch configuration.')
        for label in (self.active_label, self.draft_label, self.status_label):
            layout.addWidget(label)
        self.completed.connect(self._finished)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(3000)
        self._timer.timeout.connect(self._refresh)
        kp, kd = gain_table.DEFAULT_KP, gain_table.DEFAULT_KD
        try:
            stored = json.loads(self.store.value(gain_table.STORE_KEY, '{}'))
            if stored:
                gain_table.validate(stored['kp_ratios'], stored['kd_ratios'])
                kp, kd = stored['kp_ratios'], stored['kd_ratios']
        except (ValueError, TypeError, KeyError):
            self.status_label.setText('Saved draft invalid; defaults loaded. Read bridge before applying.')
        self._set_draft(kp, kd)

    @staticmethod
    def _label(text):
        label = QtWidgets.QLabel(text)
        label.setTextFormat(QtCore.Qt.PlainText)
        label.setWordWrap(True)
        return label

    def draft(self):
        return [w.value() for w in self.kp], [w.value() for w in self.kd]

    @property
    def busy(self):
        return self._busy

    def _set_draft(self, kp, kd):
        for widgets, values in ((self.kp, kp), (self.kd, kd)):
            for widget, value in zip(widgets, values):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self._edited()

    def _edited(self):
        if not hasattr(self, 'draft_label'):
            return
        kp, kd = self.draft()
        error = ''
        try:
            gain_table.validate(kp, kd)
        except ValueError as exc:
            error = str(exc)
        current = self._snapshot and self._snapshot.get('config')
        matches = bool(current and current['table']['kp_ratios'] == kp and
                       current['table']['kd_ratios'] == kd)
        self.draft_label.setText(error or ('Matches the confirmed bridge table.' if matches
                                          else 'Draft changes — not applied to the bridge.'))
        self.draft_label.setStyleSheet('color: #b71c1c;' if error else '')
        if not error:
            saved = json.dumps({'kp_ratios': kp, 'kd_ratios': kd})
            if self.store.value(gain_table.STORE_KEY) != saved:
                self.store.setValue(gain_table.STORE_KEY, saved)
        editable = not self._busy and not self._sending
        for widget in self.kp + self.kd + [self.side, self.defaults_button]:
            widget.setEnabled(editable)
        self.read_button.setEnabled(not self._busy)
        self.use_active_button.setEnabled(bool(editable and current))
        self.apply_button.setEnabled(bool(editable and not error and current and not matches
                                           and not self._snapshot['busy']))

    def _use_active(self):
        if self.use_active_button.isEnabled():
            table = self._snapshot['config']['table']
            self._set_draft(table['kp_ratios'], table['kd_ratios'])

    def set_output_active(self, active):
        self._sending = bool(active)
        self._edited()

    def target_changed(self, *_):
        if not hasattr(self, '_timer'):
            return
        self._token += 1
        self._cancel.set()
        self._snapshot = None
        self._busy = False
        self._timer.stop()
        self.active_label.setText('Destination changed. Read the bridge to confirm its table.')
        self.status_label.setText('No configuration sent to the new destination.')
        self._edited()
        self.busy_changed.emit(False)

    def _start(self, operation, refresh=False):
        if self._busy or self._closed:
            return
        if operation == 'apply' and not self.apply_button.isEnabled():
            return
        host, port = self.target()
        side = self.side.currentText()
        kwargs = {}
        if operation == 'apply':
            kp, kd = self.draft()
            kwargs = dict(kp=kp, kd=kd, expected_hash=self._snapshot['config']['config_hash'],
                          expected_revision=self._snapshot['config']['revision'])
        self._busy = True
        self._token += 1
        token = self._token
        self._cancel = cancel = threading.Event()
        self._edited()
        self.busy_changed.emit(True)
        if not refresh:
            self.status_label.setText('Reading bridge…' if operation == 'read' else 'Applying table…')

        def run():
            try:
                result = self.request(host, port, side, cancel=cancel, **kwargs)
            except Exception as error:
                result = {'error': str(error)}
            if not cancel.is_set():
                self.completed.emit(result, token, 'refresh' if refresh else operation)

        self._thread = threading.Thread(target=run, name='live-gain-config', daemon=True)
        self._thread.start()

    def _finished(self, result, token, operation):
        if self._closed or token != self._token:
            return
        self._busy = False
        if 'error' in result or result.get('config') is None:
            self._snapshot = None
            self.active_label.setText('Bridge table unconfirmed. Read again before applying.')
            self.status_label.setText(result.get('error') or result.get('message', 'No bridge reply'))
        else:
            previous_revision = (self._snapshot or {}).get('config', {}).get('revision')
            self._snapshot = result
            config, table = result['config'], result['config']['table']
            kp = ', '.join(f'{v:g}' for v in table['kp_ratios'])
            kd = ', '.join(f'{v:g}' for v in table['kd_ratios'])
            state = 'armed' if result['armed'] else 'busy' if result['busy'] else 'disarmed'
            self.active_label.setText(f"Confirmed {config['side']} bridge ({state})\n"
                                      f"Kp: {kp}\nKd: {kd}\nConfig: {config['config_hash']}")
            if result['success'] and operation == 'apply':
                self._set_draft(table['kp_ratios'], table['kd_ratios'])
            if operation != 'refresh' or not result['success']:
                self.status_label.setText(result['message'])
            elif previous_revision is not None and previous_revision != config['revision']:
                self.status_label.setText('Bridge table changed. Review the active values and your draft.')
            self._timer.start()
        self._edited()
        self.busy_changed.emit(False)

    def _refresh(self):
        self._start('read', refresh=True)

    def close_client(self):
        self._closed = True
        self._timer.stop()
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=.3)
