package com.eventplanning.ui

import com.eventplanning.domain.EventManager
import javax.swing.*
import javax.swing.border.EmptyBorder
import java.awt.*

class StatisticsPanel(private val eventManager: EventManager) : JPanel() {

    private val headerLabel = UIStyles.createHeaderLabel("Dashboard Overview")
    private val cardTitles = mutableListOf<JLabel>()
    private val totalEventsLabel = createStatLabel()
    private val totalParticipantsLabel = createStatLabel()
    private val totalVenuesLabel = createStatLabel()
    private val busyVenueLabel = createStatLabel()
    private val avgOccupancyLabel = createStatLabel()
    private val refreshBtn = JButton("Refresh Data")

    /**
     * Use a BorderLayout with spacing
     * Top: Header with title and refresh button
     * Center: GridLayout 2x3 with statistic cards
     */
    init {
        layout = BorderLayout(0, 30)
        isOpaque = false

        val headerPanel = JPanel(BorderLayout())
        headerPanel.isOpaque = false
        headerPanel.add(headerLabel, BorderLayout.WEST)

        refreshBtn.font = Font("Segue UI", Font.PLAIN, 14)
        refreshBtn.isFocusPainted = false
        refreshBtn.addActionListener { refreshStats() }
        headerPanel.add(refreshBtn, BorderLayout.EAST)

        add(headerPanel, BorderLayout.NORTH)

        val gridPanel = JPanel(GridLayout(2, 3, 25, 25))
        gridPanel.isOpaque = false

        /**
         * Create statistic cards and add them to the grid
         * Each card has a title, value label, and accent color
         */
        gridPanel.add(createCard("Total Events", totalEventsLabel, UIStyles.accentBlue))
        gridPanel.add(createCard("Participants", totalParticipantsLabel, UIStyles.accentGreen))
        gridPanel.add(createCard("Total Venues", totalVenuesLabel, UIStyles.accentOrange))
        gridPanel.add(createCard("Busiest Venue", busyVenueLabel, UIStyles.accentPink))
        gridPanel.add(createCard("Avg. Occupancy", avgOccupancyLabel, UIStyles.accentPurple))
        gridPanel.add(createBrandCard())

        /**
         * Add the grid panel to the center of the main panel
         */
        add(gridPanel, BorderLayout.CENTER)

        applyTheme()
        refreshStats()
    }

    /**
     * Called when the theme is changed to update colors
     * method updates the foreground colors of labels and buttons
     * and repaints the panel to reflect the new theme.
     */
    fun applyTheme() {
        headerLabel.foreground = UIStyles.textPrimary

        cardTitles.forEach { it.foreground = UIStyles.textSecondary }

        val valueLabels = listOf(totalEventsLabel, totalParticipantsLabel, totalVenuesLabel, busyVenueLabel)
        valueLabels.forEach { it.foreground = UIStyles.textPrimary }

        refreshBtn.background = UIStyles.cardBackground
        refreshBtn.foreground = UIStyles.textPrimary

        this.repaint()
        /**
         * refreshStats is called to ensure consistent styling
         */
        refreshStats()
    }

    /**
     * Creates a statistic card panel with a title, value label, and accent color.
     * The card has a custom paintComponent method to draw rounded corners and an accent bar.
     */
    private fun createCard(title: String, valueLabel: JLabel, accentColor: Color): JPanel {
        val card = object : JPanel(BorderLayout(0, 15)) {
            override fun paintComponent(g: Graphics) {
                val g2 = g as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

                /**
                 * Draw the card background with rounded corners
                 */
                g2.color = UIStyles.cardBackground
                g2.fillRoundRect(0, 0, width, height, 20, 20)
                /**
                 * Draw the accent bar at the top of the card
                 */
                g2.color = accentColor
                g2.fillRoundRect(20, 0, 60, 6, 4, 4)
            }
        }
        card.isOpaque = false
        card.border = EmptyBorder(30, 30, 30, 30)

        val lblTitle = JLabel(title)
        lblTitle.font = Font("Segue UI", Font.PLAIN, 14)
        cardTitles.add(lblTitle)

        /**
         * placed the title label at the top (NORTH) and the value label in the center (CENTER)
         */
        card.add(lblTitle, BorderLayout.NORTH)
        valueLabel.horizontalAlignment = SwingConstants.LEFT
        card.add(valueLabel, BorderLayout.CENTER)

        return card
    }

    /**
     *creates a branded card panel displaying "Event Planner Beta"
     * does not contain any actual statistics
     */
    private fun createBrandCard(): JPanel {
        return object : JPanel(GridBagLayout()) {
            override fun paintComponent(g: Graphics) {
                val g2 = g as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = UIStyles.cardBackground
                g2.fillRoundRect(0, 0, width, height, 20, 20)
            }
        }.apply {
            isOpaque = false
            val lbl = JLabel("Event Planner Beta")
            lbl.font = Font("Segue UI", Font.BOLD, 18)
            lbl.foreground = Color(100, 100, 100)
            add(lbl)
        }
    }

    /**
     * Creates a JLabel for displaying statistic values with large bold font.
     */
    private fun createStatLabel(): JLabel {
        return JLabel("...").apply {
            font = Font("Segue UI", Font.BOLD, 36)
        }
    }

    /**
     * Fetches data from the EventManager and updates the statistic labels.
     * takes gets all events, participants, and venues,
     */
    fun refreshStats() {
        val events = eventManager.getAllEvents()
        val participants = eventManager.getAllParticipants()
        val venues = eventManager.getAllVenues()

        /**
         * simple statistics like total counts are calculated directly
         */
        totalEventsLabel.text = events.size.toString()
        totalParticipantsLabel.text = participants.size.toString()
        totalVenuesLabel.text = venues.size.toString()
        /**
         * more complex statistics like busiest venue and average occupancy
         * are calculated using grouping and summation
         */
        if (events.isNotEmpty()) {
            val venuesMap = events.groupingBy { it.venue.name }.eachCount()
            val maxEntry = venuesMap.maxByOrNull { it.value }
            busyVenueLabel.text = maxEntry?.key ?: "N/A"
            busyVenueLabel.font = if ((maxEntry?.key?.length ?: 0) > 12) Font("Segue UI", Font.BOLD, 22) else Font("Segue UI", Font.BOLD, 36)
        } else {
            busyVenueLabel.text = "N/A"
        }
        /**
         * average occupancy is calculated as total registered participants
         * divided by total capacity across all events
         */
        if (events.isNotEmpty()) {
            val totalCap = events.sumOf { it.maxParticipants }
            val totalReg = events.sumOf { it.getCurrentCapacity() }
            val percent = if (totalCap > 0) (totalReg.toDouble() / totalCap * 100).toInt() else 0
            avgOccupancyLabel.text = "$percent%"

            /**
             * the color of the average occupancy label changes based on thresholds
             * >80% = green, <20% = orange, else primary text color
             */
            avgOccupancyLabel.foreground = when {
                percent > 80 -> UIStyles.accentGreen
                percent < 20 -> UIStyles.accentOrange
                else -> UIStyles.textPrimary
            }

            /**
             * if there are no events, set occupancy to 0% with primary text color
             */
        } else {
            avgOccupancyLabel.text = "0%"
            avgOccupancyLabel.foreground = UIStyles.textPrimary
        }
    }
}