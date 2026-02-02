package com.eventplanning.ui

import com.eventplanning.domain.EventManager
import com.eventplanning.domain.Participant
import javax.swing.*
import javax.swing.border.EmptyBorder
import javax.swing.table.DefaultTableModel
import javax.swing.table.TableRowSorter
import java.awt.*
import java.util.UUID

class ParticipantPanel(private val eventManager: EventManager) : JPanel() {

    private var displayedParticipants: List<Participant> = emptyList()
    private val tableModel = object : DefaultTableModel(arrayOf("Name", "Email", "Organization", "Phone"), 0) { override fun isCellEditable(row: Int, column: Int) = false }
    private val participantTable = JTable(tableModel)
    private val tableSorter = TableRowSorter(tableModel)

    private val searchField = UIStyles.createTextField(20)
    private val nameField = UIStyles.createTextField()
    private val emailField = UIStyles.createTextField()
    private val phoneField = UIStyles.createTextField()
    private val orgField = UIStyles.createTextField()

    private val formLabels = mutableListOf<JLabel>()
    private val headerLabel = UIStyles.createHeaderLabel("Participants")
    private val sectionList = UIStyles.createSectionLabel("DIRECTORY")
    private val sectionAdd = UIStyles.createSectionLabel("REGISTER NEW")
    private val searchLabel = UIStyles.createLabel("Search:")

    private val addButton = UIStyles.createPrimaryButton("Add Person")
    private val deleteButton = UIStyles.createDangerButton("Delete")

    /**
     * set up layout:
     * build the table card (left) and form card (right)
     * apply theme
     * loads participant data
     * setup listeners for buttons and search
     */
    init {
        layout = BorderLayout(0, 20)
        isOpaque = false

        add(headerLabel, BorderLayout.NORTH)
        val content = JPanel(GridBagLayout()); content.isOpaque = false

        /**
         * create table card (left) and form card (right)
         */
        val gbc = GridBagConstraints().apply{
            fill = GridBagConstraints.BOTH
            insets = Insets(0, 0, 0, 20)
        }

        /**
         * Left: Table Card for participant list
         */
        gbc.gridx = 0; gbc.weightx = 0.65
        gbc.weighty = 1.0
        content.add(createTableCard(), gbc)

        /**
         * Right: Form Card for adding participant
         */
        gbc.gridx = 1
        gbc.weightx = 0.35
        gbc.insets = Insets(0, 0, 0, 0)
        content.add(createFormCard(), gbc)

        add(content, BorderLayout.CENTER)
        /**
         * Apply theme to all components
         * Refresh table to load data
         * Setup listeners for buttons and search
         */
        applyTheme()
        refreshTable()
        setupListeners()
    }

    /**
     *called whenever the theme is changed to update colors
     * method updates the foreground colors of labels, inputs, and table
     * and repaints the panel to reflect the new theme.
     */
    fun applyTheme() {
        headerLabel.foreground = UIStyles.textPrimary
        sectionList.foreground = UIStyles.textMuted
        sectionAdd.foreground = UIStyles.textMuted
        searchLabel.foreground = UIStyles.textSecondary

        formLabels.forEach { it.foreground = UIStyles.textSecondary }
        /**
         * Update Inputs, style each input field
         */
        val inputs = listOf(searchField, nameField, emailField, phoneField, orgField)
        inputs.forEach {
            it.background = UIStyles.inputBackground
            it.foreground = UIStyles.textPrimary
            it.border = BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(UIStyles.tableBorder),
                EmptyBorder(8, 10, 8, 10)
            )
        }
        UIStyles.styleTable(participantTable)

        /**
         * repaint the panel to reflect new theme
         */
        this.repaint()
    }

    /**
     * Build the left card containing
     * Section title
     * search bar
     * participant table
     * delete button
     */
    private fun createTableCard(): JPanel {
        val card = UIStyles.createCardPanel()

        val top = JPanel(BorderLayout())
        top.isOpaque = false
        top.add(sectionList, BorderLayout.WEST)

        val searchPanel = JPanel(FlowLayout(FlowLayout.RIGHT))
        searchPanel.isOpaque = false
        searchPanel.add(searchLabel)
        searchPanel.add(searchField)
        top.add(searchPanel, BorderLayout.EAST)
        card.add(top, BorderLayout.NORTH)

        UIStyles.styleTable(participantTable)
        participantTable.rowSorter = tableSorter

        card.add(UIStyles.createScrollPane(participantTable), BorderLayout.CENTER)

        val bot = JPanel(FlowLayout(FlowLayout.LEFT))
        bot.isOpaque = false
        bot.add(deleteButton)
        card.add(bot, BorderLayout.SOUTH)

        return card
    }

    /**
     * Builds the right card containing:
     * Section title "REGISTER NEW"
     * Input fields for Full Name, Email, Phone, Organization
     * Add Person button
     */
    private fun createFormCard(): JPanel {
        val card = UIStyles.createCardPanel()
        card.add(sectionAdd, BorderLayout.NORTH)

        val form = JPanel(GridBagLayout())
        form.isOpaque = false
        val gbc = GridBagConstraints().apply {
            fill = GridBagConstraints.HORIZONTAL
            insets = Insets(0, 0, 15, 0)
            weightx = 1.0
            gridx = 0
        }
        var y = 0

        /**
         * add a label and component to the form
         */
        fun addIn(text: String, comp: JComponent) {
            gbc.gridy = y++
            val lbl = UIStyles.createLabel(text)
            formLabels.add(lbl)
            form.add(lbl, gbc)

            gbc.gridy = y++
            form.add(comp, gbc)
        }
        /**
         * Add input fields to the form
         */
        addIn("Full Name", nameField)
        addIn("Email Address", emailField)
        addIn("Phone Number", phoneField)
        addIn("Organization", orgField)

        /**
         * Spacer to push button to bottom
         */
        gbc.gridy = y++
        gbc.weighty = 1.0
        form.add(JPanel().apply { isOpaque = false }, gbc)

        /**
         * Add Person button aligned to right
         */
        gbc.gridy = y++
        gbc.weighty = 0.0
        val btnBox = JPanel(FlowLayout(FlowLayout.RIGHT))
        btnBox.isOpaque = false
        btnBox.add(addButton)
        form.add(btnBox, gbc)

        card.add(form, BorderLayout.CENTER)
        return card
    }

    /**
     * setup action listeners for:
     * add Person button
     * delete button
     * live search field (updates filter as the user types)
     */
    private fun setupListeners() {

        /**
         * Add new participant when Add Person button is clicked
         */
        addButton.addActionListener {

            /**
             * basic validation: name and email cannot be blank
             * */
            if (nameField.text.isBlank() || emailField.text.isBlank()) return@addActionListener

            /**
             * create a participant object with a random UUID
             */
            val p = Participant(
                UUID.randomUUID().toString(),
                nameField.text,
                emailField.text,
                phoneField.text,
                orgField.text
            )

            val worker = object : SwingWorker<Boolean, Void>() {
                override fun doInBackground() = eventManager.addParticipant(p)

                /**
                 * if addition was successful
                 * refresh table and clear input fields
                 */
                override fun done() {
                    if (get()) {
                        refreshTable()
                        nameField.text = ""
                        emailField.text = ""
                        phoneField.text = ""
                        orgField.text = ""
                    }
                }
            }
            worker.execute()
        }

        /**
         * Delete selected participant when Delete button is clicked
         * also converts view index to model index for correct selection
         * and finds the corresponding Participant in displayedParticipants to delete
         */
        deleteButton.addActionListener {
            val row = participantTable.selectedRow
            if (row != -1) {
                val modelIndex = participantTable.convertRowIndexToModel(row)
                val p = displayedParticipants[modelIndex]

                val worker = object : SwingWorker<Boolean, Void>() {
                    override fun doInBackground() = eventManager.deleteParticipant(p)
                    override fun done() {
                        if (get()) {

                            /**
                             * refresh table after deletion
                             */
                            refreshTable()
                        }
                    }
                }
                worker.execute()
            }
        }

        /**
         * Live search filter as user types in search field
         * uses case-insensitive regex filter on all columns
         */
        searchField.document.addDocumentListener(object : javax.swing.event.DocumentListener {
            override fun insertUpdate(e: javax.swing.event.DocumentEvent?) = filter()
            override fun removeUpdate(e: javax.swing.event.DocumentEvent?) = filter()
            override fun changedUpdate(e: javax.swing.event.DocumentEvent?) = filter()

            fun filter() {

                /**
                 * if search field is blank, remove filter
                 * else apply regex filter with case-insensitivity
                 */
                val text = searchField.text
                if (text.isBlank()) {
                    tableSorter.rowFilter = null
                } else {
                    tableSorter.rowFilter = RowFilter.regexFilter("(?i)$text")
                }
            }
        })
    }

    /**
     * reloads the list of participants from EventManager
     * updates displayedParticipants so selection/deletion can work correctly.
     */
    private fun refreshTable() {
        tableModel.rowCount = 0

        displayedParticipants = eventManager.getAllParticipants()

        displayedParticipants.forEach {
            tableModel.addRow(arrayOf(it.name, it.email, it.organization, it.phone))
        }
    }
}