package com.eventplanning.ui

import com.eventplanning.domain.Event
import com.eventplanning.domain.EventManager
import com.eventplanning.domain.Venue
import com.eventplanning.service.ScalaBridge
import javax.swing.*
import javax.swing.border.EmptyBorder
import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener
import javax.swing.table.DefaultTableCellRenderer
import javax.swing.table.DefaultTableModel
import javax.swing.table.TableRowSorter
import java.awt.*
import java.time.Duration
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Date
import java.util.Calendar
import java.util.UUID

class EventPanel(private val eventManager: EventManager) : JPanel() {

    private var displayedEvents: List<Event> = emptyList()
    private var editingEventId: String? = null

    private val tableModel = object : DefaultTableModel(arrayOf("Title", "Date", "Time", "Duration", "Venue", "Occupancy"), 0) {
        override fun isCellEditable(row: Int, column: Int) = false
    }
    private val eventTable = JTable(tableModel)
    private val tableSorter = TableRowSorter(tableModel)

    private val headerLabel = UIStyles.createHeaderLabel("Events Management")
    private val sectionSchedule = UIStyles.createSectionLabel("SCHEDULE")
    private val sectionDetails = UIStyles.createSectionLabel("EVENT DETAILS")

    private val lblTitle = UIStyles.createLabel("Event Title")
    private val lblDate = UIStyles.createLabel("Start Date & Time")
    private val lblDur = UIStyles.createLabel("Duration")
    private val lblVen = UIStyles.createLabel("Venue")
    private val lblCap = UIStyles.createLabel("Max Capacity")
    private val lblDesc = UIStyles.createLabel("Description")

    private val searchField = UIStyles.createTextField(15).apply {
        putClientProperty("JTextField.placeholderText", "Search...")
    }
    private val titleField = UIStyles.createTextField()
    private val venueCombo = UIStyles.createComboBox()
    private val descriptionArea = UIStyles.createTextArea(3, 20)
    private val maxParticipantsField = UIStyles.createTextField(10)

    private val dateSpinner = JSpinner(SpinnerDateModel().apply { calendarField = Calendar.DAY_OF_MONTH }).apply { editor = JSpinner.DateEditor(this, "dd/MM/yyyy") }
    private val timeSpinner = JSpinner(SpinnerDateModel().apply { calendarField = Calendar.MINUTE }).apply { editor = JSpinner.DateEditor(this, "HH:mm") }
    private val hoursSpinner = JSpinner(SpinnerNumberModel(2, 0, 24, 1))
    private val minutesSpinner = JSpinner(SpinnerNumberModel(0, 0, 59, 15))

    private val saveButton = UIStyles.createPrimaryButton("Save Event")
    private val clearButton = UIStyles.createSecondaryButton("Cancel")
    private val editButton = UIStyles.createAccentButton("Edit Selected")
    private val deleteButton = UIStyles.createDangerButton("Delete")
    private val findVenueBtn = UIStyles.createSecondaryButton("Find Slot")

    /**
     * sets up
     * Header at the top
     * left card with event table, search, edit, delete
     * right card with event form fields and save/cancel buttons and find slot button
     * applies theming and listeners
     */
    init {
        layout = BorderLayout(0, 20)
        isOpaque = false

        add(headerLabel, BorderLayout.NORTH)

        val content = JPanel(GridBagLayout())
        content.isOpaque = false
        val gbc = GridBagConstraints().apply { fill = GridBagConstraints.BOTH; insets = Insets(0, 0, 0, 20) }

        gbc.gridx = 0; gbc.weightx = 0.65; gbc.weighty = 1.0
        content.add(createTableCard(), gbc)

        gbc.gridx = 1; gbc.weightx = 0.35; gbc.insets = Insets(0, 0, 0, 0)
        content.add(createFormCard(), gbc)

        add(content, BorderLayout.CENTER)

        applyTheme()
        refreshEventTable()
        refreshVenueCombo()
        setupListeners()
    }

    /**
     * Called when the theme is changed to update colors
     * method updates the foreground colors of labels, inputs, table, and buttons
     * and repaints the panel to reflect the new theme.
     */
    fun applyTheme() {
        headerLabel.foreground = UIStyles.textPrimary
        sectionSchedule.foreground = UIStyles.textMuted
        sectionDetails.foreground = UIStyles.textMuted

        val labels = listOf(lblTitle, lblDate, lblDur, lblVen, lblCap, lblDesc)
        labels.forEach { it.foreground = UIStyles.textSecondary }

        val inputs = listOf(titleField, searchField, maxParticipantsField)
        inputs.forEach {
            it.background = UIStyles.inputBackground
            it.foreground = UIStyles.textPrimary
            it.border = BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(UIStyles.tableBorder), EmptyBorder(8,10,8,10))
        }

        /**
         * styles the description text area
         * sets background and foreground colors for consistent theming
         */
        descriptionArea.background = UIStyles.inputBackground
        descriptionArea.foreground = UIStyles.textPrimary

        /**
         * styles the venue combo box
         * using UIStyles.styleComboBox for consistent theming
         */
        UIStyles.styleComboBox(venueCombo)

        /**
         * styles the date/time/duration spinners
         * sets background and foreground colors for consistent theming
         */
        val spinners = listOf(dateSpinner, timeSpinner, hoursSpinner, minutesSpinner)
        spinners.forEach { s ->
            s.background = UIStyles.inputBackground
            s.border = BorderFactory.createLineBorder(UIStyles.tableBorder)
            val textField = when (val editor = s.editor) {
                is JSpinner.DefaultEditor -> editor.textField
                is JSpinner.DateEditor -> editor.textField
                else -> null
            }
            textField?.background = UIStyles.inputBackground
            textField?.foreground = UIStyles.textPrimary
        }

        /**
         * styles the event table
         * using UIStyles.styleTable for consistent theming
         */
        UIStyles.styleTable(eventTable)
        eventTable.columnModel.getColumn(5).cellRenderer = CapacityCellRenderer()

        this.revalidate()
        this.repaint()
    }

    /**
     * creates the left card panel containing
     * section title
     * scrollable event table
     * search, edit and delete buttons
     */
    private fun createTableCard(): JPanel {
        val card = UIStyles.createCardPanel()

        val header = JPanel(BorderLayout())
        header.isOpaque = false
        header.add(sectionSchedule, BorderLayout.WEST)

        val tools = JPanel(FlowLayout(FlowLayout.RIGHT, 5, 0))
        tools.isOpaque = false
        tools.add(searchField); tools.add(editButton); tools.add(deleteButton)
        header.add(tools, BorderLayout.EAST)

        card.add(header, BorderLayout.NORTH)
        card.add(UIStyles.createScrollPane(eventTable), BorderLayout.CENTER)
        return card
    }

    /**
     * creates the right card panel containing
     * event form fields (title, date/time, duration, venue, capacity, description)
     * find slot button
     * save and cancel buttons
     */
    private fun createFormCard(): JPanel {
        val card = UIStyles.createCardPanel()
        card.add(sectionDetails, BorderLayout.NORTH)

        val form = JPanel(GridBagLayout())
        form.isOpaque = false
        val gbc = GridBagConstraints().apply { fill = GridBagConstraints.HORIZONTAL; insets = Insets(0, 0, 12, 0); weightx = 1.0; gridx = 0 }
        var y = 0

        /**
         * helper function to add label and component pairs to the form
         */
        fun addIn(label: JLabel, comp: JComponent) {
            gbc.gridy = y++; form.add(label, gbc)
            gbc.gridy = y++; form.add(comp, gbc)
        }

        addIn(lblTitle, titleField)

        /**
         * creates a panel for date and time spinners side by side
         */
        val dtPanel = JPanel(FlowLayout(FlowLayout.LEFT, 0, 0)); dtPanel.isOpaque = false
        dateSpinner.preferredSize = Dimension(130, 36)
        timeSpinner.preferredSize = Dimension(90, 36)
        dtPanel.add(dateSpinner); dtPanel.add(Box.createHorizontalStrut(10)); dtPanel.add(timeSpinner)
        addIn(lblDate, dtPanel)

        /**
         * creates a panel for duration spinners (hours and minutes) side by side
         */
        val dPanel = JPanel(FlowLayout(FlowLayout.LEFT, 0, 0)); dPanel.isOpaque = false
        hoursSpinner.preferredSize = Dimension(70, 36); minutesSpinner.preferredSize = Dimension(70, 36)
        dPanel.add(hoursSpinner); dPanel.add(UIStyles.createLabel(" h ")); dPanel.add(minutesSpinner); dPanel.add(UIStyles.createLabel(" m"))
        addIn(lblDur, dPanel)

        /**
         * creates a panel for venue combo box and find slot button side by side
         */
        val vPanel = JPanel(BorderLayout(5, 0)); vPanel.isOpaque = false
        vPanel.add(venueCombo, BorderLayout.CENTER); vPanel.add(findVenueBtn, BorderLayout.EAST)
        addIn(lblVen, vPanel)

        addIn(lblCap, maxParticipantsField)
        addIn(lblDesc, UIStyles.createScrollPane(descriptionArea))

        gbc.gridy = y++; gbc.weighty = 1.0; form.add(JPanel().apply{isOpaque=false}, gbc)

        /**
         * creates a panel for save and cancel buttons aligned to the right
         */
        val bPanel = JPanel(FlowLayout(FlowLayout.RIGHT)); bPanel.isOpaque = false
        bPanel.add(clearButton); bPanel.add(saveButton)
        gbc.gridy = y++; gbc.weighty = 0.0; form.add(bPanel, gbc)

        card.add(form, BorderLayout.CENTER)
        return card
    }


    /**
     * set up action listeners to
     * save/cancel event on save button click
     * edit/delete selected event on respective button clicks
     * find slot button (scala slot finder)
     * search field filtering live
     */
    private fun setupListeners() {
        saveButton.addActionListener { if (editingEventId == null) createEvent() else updateEvent() }
        clearButton.addActionListener { clearFields() }
        editButton.addActionListener { loadEventForEditing() }
        deleteButton.addActionListener { deleteSelectedEvent() }
        findVenueBtn.addActionListener { findAvailableVenue() }
        searchField.document.addDocumentListener(object : DocumentListener {
            /**
             * filters the event table based on search field input
             */
            override fun insertUpdate(e: DocumentEvent?) = filter()
            override fun removeUpdate(e: DocumentEvent?) = filter()
            override fun changedUpdate(e: DocumentEvent?) = filter()
            fun filter() {
                val text = searchField.text
                /**
                 * if search field is empty, show all rows
                 * else filter rows matching the text
                 */
                if (text.trim().isEmpty()) tableSorter.rowFilter = null else tableSorter.rowFilter = RowFilter.regexFilter("(?i)$text")
            }
        })
    }

    /**
     * combine the date and time spinners into a single LocalDateTime instance
     */
    private fun getCombinedDateTime(): LocalDateTime {
        val datePart = (dateSpinner.value as Date).toInstant().atZone(ZoneId.systemDefault()).toLocalDate()
        val timePart = (timeSpinner.value as Date).toInstant().atZone(ZoneId.systemDefault()).toLocalTime()
        return LocalDateTime.of(datePart, timePart)
    }

    /**
     * sets the date and time spinners from a given LocalDateTime instance
     */
    private fun setCombinedDateTime(dt: LocalDateTime) {
        val date = Date.from(dt.atZone(ZoneId.systemDefault()).toInstant())
        dateSpinner.value = date
        timeSpinner.value = date
    }

    /**
     * creates a new event from form fields and saves it via event manager
     * performs validation on required fields
     * uses duration and date/time spinners to construct event
     * shows success or error messages based on result
     */
    private fun createEvent() {
        val title = titleField.text.trim()
        val venueItem = venueCombo.selectedItem as? VenueItem
        val description = descriptionArea.text.trim()
        val maxParticipants = maxParticipantsField.text.toIntOrNull()

        if (title.isBlank() || venueItem == null || maxParticipants == null) {
            JOptionPane.showMessageDialog(this, "Title, Venue, and Capacity are required.", "Error", JOptionPane.WARNING_MESSAGE)
            return
        }

        val hours = (hoursSpinner.value as Number).toLong()
        val minutes = (minutesSpinner.value as Number).toLong()
        val duration = Duration.ofHours(hours).plusMinutes(minutes)
        val dateTime = getCombinedDateTime()

        /**
         * creates the Event instance with a new UUID
         * using the collected form data
         */
        val event = Event(
            id = UUID.randomUUID().toString(),
            title = title,
            dateTime = dateTime,
            venue = venueItem.venue,
            description = description,
            duration = duration,
            maxParticipants = maxParticipants
        )

        /**
         * uses a SwingWorker to add the event in the background
         * to avoid blocking the UI thread
         * on success, refreshes the table and clears fields
         */
        val worker = object : SwingWorker<Boolean, Void>() {
            override fun doInBackground(): Boolean = eventManager.addEvent(event)
            override fun done() {
                if (get()) {
                    refreshEventTable()
                    clearFields()
                    JOptionPane.showMessageDialog(this@EventPanel, "Event created!", "Success", JOptionPane.INFORMATION_MESSAGE)
                } else {
                    JOptionPane.showMessageDialog(this@EventPanel, "Failed to create event (Conflict or Capacity).", "Error", JOptionPane.ERROR_MESSAGE)
                }
            }
        }
        worker.execute()
    }

    /**
     * updates an existing event being edited
     * use the editingEventId to fetch the old event
     * build an updated event from new form values
     * copies over registered participants from old event
     * persists the updated event via eventManager.modifyEvent
     * shows success or error messages based on result
     */
    private fun updateEvent() {
        val id = editingEventId ?: return
        val oldEvent = eventManager.getEventById(id) ?: return

        val title = titleField.text.trim()
        val venueItem = venueCombo.selectedItem as? VenueItem ?: return
        val description = descriptionArea.text.trim()
        val maxParticipants = maxParticipantsField.text.toIntOrNull() ?: oldEvent.maxParticipants

        val hours = (hoursSpinner.value as Number).toLong()
        val minutes = (minutesSpinner.value as Number).toLong()
        val duration = Duration.ofHours(hours).plusMinutes(minutes)
        val dateTime = getCombinedDateTime()

        val updatedEvent = Event(
            id = id,
            title = title,
            dateTime = dateTime,
            venue = venueItem.venue,
            description = description,
            duration = duration,
            maxParticipants = maxParticipants
        )

        oldEvent.getRegisteredParticipants().forEach { updatedEvent.registerParticipant(it) }

        val worker = object : SwingWorker<Boolean, Void>() {
            override fun doInBackground(): Boolean = eventManager.modifyEvent(updatedEvent)
            override fun done() {
                if (get()) {
                    refreshEventTable()
                    clearFields()
                    JOptionPane.showMessageDialog(this@EventPanel, "Event updated!", "Success", JOptionPane.INFORMATION_MESSAGE)
                } else {
                    JOptionPane.showMessageDialog(this@EventPanel, "Update failed.", "Error", JOptionPane.ERROR_MESSAGE)
                }
            }
        }
        worker.execute()
    }

    /**
     * deletes the selected event from the table
     * confirms deletion with the user
     * uses eventManager.deleteEvent to remove it
     * refreshes the table on success
     */
    private fun deleteSelectedEvent() {
        val row = eventTable.selectedRow
        if (row == -1) return
        val event = displayedEvents.getOrNull(eventTable.convertRowIndexToModel(row)) ?: return

        if (JOptionPane.showConfirmDialog(this, "Delete '${event.title}'?", "Confirm", JOptionPane.YES_NO_OPTION) == JOptionPane.YES_OPTION) {
            val worker = object : SwingWorker<Boolean, Void>() {
                override fun doInBackground() = eventManager.deleteEvent(event)
                override fun done() {
                    if (get()) refreshEventTable() else JOptionPane.showMessageDialog(this@EventPanel, "Delete failed.", "Error", JOptionPane.ERROR_MESSAGE)
                }
            }
            worker.execute()
        }
    }

    /**
     * loads the selected event from the table into the form for editing
     * populates all form fields with event data
     * changes save button text to "Update Event"
     */
    private fun loadEventForEditing() {
        val row = eventTable.selectedRow
        if (row == -1) return
        val event = displayedEvents.getOrNull(eventTable.convertRowIndexToModel(row)) ?: return

        editingEventId = event.id
        saveButton.text = "Update Event"
        titleField.text = event.title
        descriptionArea.text = event.description
        maxParticipantsField.text = event.maxParticipants.toString()

        setCombinedDateTime(event.dateTime)
        hoursSpinner.value = event.duration.toHours().toInt()
        minutesSpinner.value = event.duration.toMinutesPart()

        for (i in 0 until venueCombo.itemCount) {
            val item = venueCombo.getItemAt(i) as VenueItem
            if (item.venue.id == event.venue.id) {
                venueCombo.selectedIndex = i
                break
            }
        }
    }

    /**
     * calls the Scala slot finder to find available venues and time slots
     * uses current capacity, date/time, and duration from form fields
     * shows a list of available slots or error messages based on result
     */
    private fun findAvailableVenue() {
        try {
            val cap = maxParticipantsField.text.toIntOrNull()
            if (cap == null) {
                JOptionPane.showMessageDialog(this, "Enter capacity first.", "Info", JOptionPane.WARNING_MESSAGE)
                return
            }
            val hours = (hoursSpinner.value as Number).toLong()
            val minutes = (minutesSpinner.value as Number).toLong()
            val dur = Duration.ofHours(hours).plusMinutes(minutes)
            val dt = getCombinedDateTime()

            findVenueBtn.isEnabled = false
            findVenueBtn.text = "..."

            val worker = object : SwingWorker<ScalaBridge.SlotFinderResult, Void>() {
                override fun doInBackground() = ScalaBridge.findAvailableVenues(eventManager.getAllVenues(), eventManager.getAllEvents(), cap, dt, dur)
                override fun done() {
                    findVenueBtn.isEnabled = true
                    findVenueBtn.text = "Find Slot"
                    try {
                        when (val result = get()) {
                            is ScalaBridge.SlotFinderResult.Success -> {
                                if (result.slots.isEmpty()) JOptionPane.showMessageDialog(this@EventPanel, "No slots found.", "Info", JOptionPane.INFORMATION_MESSAGE)
                                else {
                                    val sb = StringBuilder("Available Slots:\n\n")
                                    val fmt = DateTimeFormatter.ofPattern("HH:mm")
                                    result.slots.forEach {
                                        sb.append("• ${it.venue.name} (${it.venue.capacity})\n   ${it.startTime.format(fmt)} - ${it.freeUntil.format(fmt)}\n")
                                    }
                                    val area = UIStyles.createTextArea(10, 30).apply { text = sb.toString(); isEditable = false }
                                    JOptionPane.showMessageDialog(this@EventPanel, UIStyles.createScrollPane(area), "Slots", JOptionPane.PLAIN_MESSAGE)
                                }
                            }
                            is ScalaBridge.SlotFinderResult.Error -> JOptionPane.showMessageDialog(this@EventPanel, result.message, "Error", JOptionPane.ERROR_MESSAGE)
                        }
                    } catch (e: Exception) { e.printStackTrace() }
                }
            }
            worker.execute()
        } catch (e: Exception) { e.printStackTrace() }
    }

    /**
     * reload the event from EventManager into the table
     * formats date, time and duration into readable strings
     * display occupancy as "current/max" for the custom cell renderer
     */
    private fun refreshEventTable() {
        tableModel.rowCount = 0
        val dF = DateTimeFormatter.ofPattern("yyyy-MM-dd")
        val tF = DateTimeFormatter.ofPattern("HH:mm")

        displayedEvents = eventManager.getAllEvents()

        displayedEvents.forEach { e ->
            val duration = "${e.duration.toHours()}h ${e.duration.toMinutesPart()}m"

            tableModel.addRow(
                arrayOf(
                    e.title,
                    e.dateTime.format(dF),
                    e.dateTime.format(tF),
                    duration,
                    e.venue.name,
                    "${e.getCurrentCapacity()}/${e.maxParticipants}"
                )
            )
        }
    }

    /**
     * refreshes the venue combo box with the latest venues from EventManager
     */
    private fun refreshVenueCombo() {
        venueCombo.removeAllItems()
        eventManager.getAllVenues().forEach { venueCombo.addItem(VenueItem(it)) }
    }

    /**
     * resets the form into its default state
     * clears text fields and description area
     * clears selection and editingByEventId
     * resets date/time and duration spinners to defaults
     */
    private fun clearFields() {
        titleField.text = ""
        descriptionArea.text = ""
        maxParticipantsField.text = ""
        venueCombo.selectedIndex = -1
        editingEventId = null
        saveButton.text = "Save Event"
        eventTable.clearSelection()
        val now = Date()
        dateSpinner.value = now
        timeSpinner.value = now
        hoursSpinner.value = 2
        minutesSpinner.value = 0
    }

    /**
     * wrapper class for venue items in the combo box
     * showing venue by its name in the dropdown
     */
    private data class VenueItem(val venue: Venue) { override fun toString() = venue.name }

    /**
     * custom cell renderer to display occupancy as a progress bar
     * uses JProgressBar to visually represent current/max capacity
     * colors the bar green if under capacity, red if full
     */
    private class CapacityCellRenderer : DefaultTableCellRenderer() {
        private val progressBar = JProgressBar(0, 100)
        init {
            progressBar.isStringPainted = true
            progressBar.border = BorderFactory.createEmptyBorder(2, 2, 2, 2)
        }

        /**
         * overrides the default cell rendering to show a progress bar
         */
        override fun getTableCellRendererComponent(table: JTable?, value: Any?, isSelected: Boolean, hasFocus: Boolean, row: Int, column: Int): Component {
            progressBar.background = UIStyles.cardBackground
            if (isSelected) progressBar.background = UIStyles.tableSelection

            /**
             * parses the "current/max" string to set progress bar values
             */
            val strValue = value as? String ?: "0/0"
            try {
                val parts = strValue.split("/")
                if (parts.size == 2) {
                    val c = parts[0].trim().toInt(); val m = parts[1].trim().toInt()
                    progressBar.maximum = m; progressBar.value = c
                    progressBar.string = "$c / $m"
                    progressBar.foreground = if(c >= m) UIStyles.accentRed else UIStyles.accentGreen
                    SwingUtilities.updateComponentTreeUI(progressBar)
                }

                /**
                 * catch any parsing errors silently
                 */
            } catch(_:Exception){}
            return progressBar
        }
    }
}