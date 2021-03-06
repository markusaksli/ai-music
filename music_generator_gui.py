# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'musicbotgui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import time

import music21.midi.realtime
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf

from music_transformer.convert import midi2idxenc, idxenc2stream
from music_transformer.transformer import MusicGenerator
from music_transformer.vocab import MusicVocab


class Ui_MainWindow(object):
    def __init__(self):
        model = tf.saved_model.load('trained_models/decoder_only_smaller_1024_mega_ds')
        self.generator = MusicGenerator(model)
        self.vocab = MusicVocab.create()
        self.generated = None
        self.sequence = None
        self.player = None
        self.is_generating = False
        self.is_playing = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(437, 424)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setMaximumSize(QtCore.QSize(100, 30))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setMaximumSize(QtCore.QSize(100, 30))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.gen_amount = QtWidgets.QLineEdit(self.centralwidget)
        self.gen_amount.setMaximumSize(QtCore.QSize(100, 30))
        self.gen_amount.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.gen_amount)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setMinimumSize(QtCore.QSize(100, 20))
        self.comboBox.setMaximumSize(QtCore.QSize(100, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setCurrentIndex(1)
        self.horizontalLayout_4.addWidget(self.comboBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.top_k_notes = QtWidgets.QLineEdit(self.centralwidget)
        self.top_k_notes.setMaximumSize(QtCore.QSize(100, 30))
        self.top_k_notes.setObjectName("lineEdit_5")
        self.horizontalLayout_5.addWidget(self.top_k_notes)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem5)
        self.top_k_durations = QtWidgets.QLineEdit(self.centralwidget)
        self.top_k_durations.setMaximumSize(QtCore.QSize(100, 30))
        self.top_k_durations.setObjectName("lineEdit_6")
        self.horizontalLayout_6.addWidget(self.top_k_durations)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem6)
        self.top_k_offset = QtWidgets.QLineEdit(self.centralwidget)
        self.top_k_offset.setMaximumSize(QtCore.QSize(100, 30))
        self.top_k_offset.setObjectName("lineEdit_7")
        self.horizontalLayout_7.addWidget(self.top_k_offset)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem7)
        self.beam_width = QtWidgets.QLineEdit(self.centralwidget)
        self.beam_width.setMaximumSize(QtCore.QSize(100, 30))
        self.beam_width.setObjectName("lineEdit_8")
        self.horizontalLayout_8.addWidget(self.beam_width)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_10.addWidget(self.label_9)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem8)
        self.creativity = QtWidgets.QLineEdit(self.centralwidget)
        self.creativity.setMaximumSize(QtCore.QSize(100, 30))
        self.creativity.setObjectName("lineEdit_9")
        self.horizontalLayout_10.addWidget(self.creativity)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.generate_button = QtWidgets.QPushButton(self.centralwidget)
        self.generate_button.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.generate_button)
        self.extend_button = QtWidgets.QPushButton(self.centralwidget)
        self.extend_button.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.extend_button)
        self.start_stop_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_stop_button.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.start_stop_button)
        self.open_midi_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_midi_button.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.open_midi_button)
        self.save_midi_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_midi_button.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.save_midi_button)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.generate_button.clicked.connect(self.on_generate_clicked)
        self.extend_button.clicked.connect(self.on_extend_clicked)
        self.start_stop_button.clicked.connect(self.on_stop_clicked)
        self.open_midi_button.clicked.connect(self.open_midi)
        self.save_midi_button.clicked.connect(self.save_to_file)
        self.extend_button.setDisabled(True)
        self.open_midi_button.setDisabled(True)
        self.save_midi_button.setDisabled(True)
        self.start_stop_button.setDisabled(True)

        self.gen_amount.setText('64')
        self.top_k_notes.setText('128')
        self.top_k_durations.setText('128')
        self.top_k_offset.setText('0')
        self.beam_width.setText('3')
        self.creativity.setText('0')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Music generator"))
        self.label.setText(_translate("MainWindow", "Prompt MIDI file path (leave empty to generate from nothing):"))
        self.label_2.setText(_translate("MainWindow", "Number of input notes from MIDI file (leave empty for all notes)"))
        self.label_3.setText(_translate("MainWindow", "Number of new notes to generate"))
        self.label_4.setText(_translate("MainWindow", "Search"))
        self.comboBox.setItemText(0, _translate("MainWindow", "greedy"))
        self.comboBox.setItemText(1, _translate("MainWindow", "top_k"))
        self.comboBox.setItemText(2, _translate("MainWindow", "beam"))
        self.label_5.setText(_translate("MainWindow", "top_k_notes"))
        self.label_6.setText(_translate("MainWindow", "top_k_durations"))
        self.label_7.setText(_translate("MainWindow", "top_k_offset"))
        self.label_8.setText(_translate("MainWindow", "beam_width (not recommended over 10)"))
        self.label_9.setText(_translate("MainWindow", "creativity (0-1000)"))
        self.generate_button.setText(_translate("MainWindow", "Generate new sequence"))
        self.extend_button.setText(_translate("MainWindow", "Extend current sequence"))
        self.start_stop_button.setText(_translate("MainWindow", "Stop playing"))
        self.open_midi_button.setText(_translate("MainWindow", "Open generated sequence as MIDI file"))
        self.save_midi_button.setText(_translate("MainWindow", "Save generated sequence to \'generated\' directory"))

    def set_playing_false(self):
        self.is_playing = False

    def generate(self, extend=False):
        try:
            midi_path = self.lineEdit.text()
            gen_amount = int(self.gen_amount.text()) * 2
            search_type = self.comboBox.currentText()
            top_k_notes = int(self.top_k_notes.text())
            top_k_durations = int(self.top_k_durations.text())
            top_k_offset = int(self.top_k_offset.text())
            beam_width = int(self.beam_width.text())
            creativity = int(self.creativity.text())
            if midi_path and not extend:
                inp = midi2idxenc(midi_path, self.vocab, add_eos=False)
            elif extend:
                inp = self.sequence
            else:
                inp = np.array([1, 0])

            input_amount = self.lineEdit_2.text()
            if not input_amount or extend:
                input_amount = len(inp)
            else:
                input_amount = int(input_amount) * 2
            inp = inp[:input_amount]
        except Exception as e:
            print(e)
            self.generate_button.setText('Invalid inputs')
            return

        try:
            self.generate_button.setText('Generating...')
            self.generate_button.setDisabled(True)
            generated = self.generator.extend_sequence(inp, max_generate_len=gen_amount, search=search_type,
                                                       top_k_notes=top_k_notes, top_k_durations=top_k_durations,
                                                       top_k_offset=top_k_offset, beam_width=beam_width,
                                                       creativity=creativity)
            self.sequence = generated.numpy()
            self.generated = idxenc2stream(self.sequence, vocab=self.vocab)
            self.player = music21.midi.realtime.StreamPlayer(self.generated)
            self.player.play(blocked=False, endFunction=self.set_playing_false)
            self.is_playing = True
            self.start_stop_button.setDisabled(False)
            self.start_stop_button.setText('Stop playing')
            self.generate_button.setDisabled(False)
            self.open_midi_button.setDisabled(False)
            self.save_midi_button.setDisabled(False)
            if not extend:
                self.extend_button.setDisabled(False)
            self.generate_button.setText('Generate new sequence')
        except Exception as e:
            self.generate_button.setText("Error occured")
            print(e)
        finally:
            self.generate_button.setDisabled(False)

    def on_generate_clicked(self):
        self.generate()
    def on_extend_clicked(self):
        self.generate(extend=True)

    def on_stop_clicked(self):
        if self.player is not None:
            if self.is_playing:
                self.player.stop()
                self.start_stop_button.setText('Play generated sequence')
                self.is_playing = False
            else:
                self.player.play(blocked=False, endFunction=self.set_playing_false)
                self.start_stop_button.setText('Stop playing')
                self.is_playing = True

    def open_midi(self):
        if self.generated is not None:
            self.generated.show('midi')

    def save_to_file(self):
        if self.generated is not None:
            try:
                self.save_midi_button.setText("Save generated sequence to \'generated\' directory")
                if not os.path.isdir('./generated'):
                    os.mkdir('generated')
                self.generated.write('midi', fp=f'./generated/{int(time.time())}.mid')
            except Exception as e:
                self.save_midi_button.setText("Error occured")
                print(e)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
