package com.eventplanning.ui

import java.awt.*
import java.awt.geom.RoundRectangle2D
import javax.swing.*
import javax.swing.border.EmptyBorder
import javax.swing.table.DefaultTableCellRenderer

object UIStyles {

    /**
     * Data class representing a UI theme with various color properties.
     * accent colors are predefined for consistency across themes.
     */
    data class Theme(
        val isDark: Boolean,
        val background: Color,
        val cardBackground: Color,
        val inputBackground: Color,
        val sidebarBackground: Color,
        val textPrimary: Color,
        val textSecondary: Color,
        val textMuted: Color,
        val tableBorder: Color,
        val tableSelection: Color,

        val accentGreen: Color = Color(30, 215, 96),
        val accentBlue: Color = Color(65, 105, 225),
        val accentRed: Color = Color(225, 50, 50),
        val accentOrange: Color = Color(255, 165, 0),
        val accentPurple: Color = Color(138, 43, 226),
        val accentPink: Color = Color(255, 105, 180)
    )

    /**
     * Predefined Dark and Light themes with specific color values.
     */
    private val DarkTheme = Theme(
        isDark = true,
        background = Color(18, 18, 18),
        cardBackground = Color(33, 33, 33),
        inputBackground = Color(60, 60, 60),
        sidebarBackground = Color(12, 12, 12),
        textPrimary = Color(255, 255, 255),
        textSecondary = Color(179, 179, 179),
        textMuted = Color(110, 110, 110),
        tableBorder = Color(50, 50, 50),
        tableSelection = Color(50, 50, 50)
    )

    /**
     * predefined Light theme with specific color values.
     */
    private val LightTheme = Theme(
        isDark = false,
        background = Color(240, 242, 245),
        cardBackground = Color(255, 255, 255),
        inputBackground = Color(255, 255, 255),
        sidebarBackground = Color(12, 12, 12),
        textPrimary = Color(30, 30, 30),
        textSecondary = Color(80, 80, 80),
        textMuted = Color(120, 120, 120),
        tableBorder = Color(220, 220, 220),
        tableSelection = Color(230, 240, 255)
    )

    /**
     * Current theme in use, initialized to DarkTheme.
     */
    var current: Theme = DarkTheme
        private set

    /**
     * List of listeners to notify when the theme changes.
     */
    private val listeners = mutableListOf<() -> Unit>()

    /**
     * Registers a listener to be notified when the theme changes.
     */
    fun addThemeListener(listener: () -> Unit) {
        listeners.add(listener)
    }

    /**
     * Toggles between Dark and Light themes and notifies listeners.
     */
    fun toggleTheme() {
        current = if (current == DarkTheme) LightTheme else DarkTheme
        listeners.forEach { it() }
    }

    /**
     * Exposed color properties for easy access to the current theme's colors.
     * Panels uses these properties to style components consistently.
     */
    val background get() = current.background
    val cardBackground get() = current.cardBackground
    val inputBackground get() = current.inputBackground
    val sidebarBackground get() = current.sidebarBackground
    val textPrimary get() = current.textPrimary
    val textSecondary get() = current.textSecondary
    val textMuted get() = current.textMuted
    val tableBorder get() = current.tableBorder
    val tableSelection get() = current.tableSelection

    val accentGreen get() = current.accentGreen
    val accentBlue get() = current.accentBlue
    val accentRed get() = current.accentRed
    val accentOrange get() = current.accentOrange
    val accentPurple get() = current.accentPurple
    val accentPink get() = current.accentPink

    val fontHeader = Font("Segue UI", Font.BOLD, 26)
    val fontSection = Font("Segue UI", Font.BOLD, 12)
    val fontBody = Font("Segue UI", Font.PLAIN, 14)
    val fontBold = Font("Segue UI", Font.BOLD, 14)


    /**
     * create a large header at the top of panels
     * Events, Venues, Participants, Statistics
     */
    fun createHeaderLabel(text: String): JLabel {
        return JLabel(text).apply {
            font = fontHeader
            foreground = textPrimary
            border = EmptyBorder(0, 0, 20, 0)
        }
    }

    /**
     * create a section label to separate different sections in forms
     * DIRECTORY, REGISTER NEW
     */
    fun createSectionLabel(text: String): JLabel {
        return JLabel(text.uppercase()).apply {
            font = fontSection
            foreground = textMuted
            border = EmptyBorder(15, 0, 10, 0)
        }
    }

    /**
     * create a standard label for form fields
     */
    fun createLabel(text: String): JLabel {
        return JLabel(text).apply {
            font = fontBody
            foreground = textSecondary
        }
    }

    /**
     * create a styled text field for user input
     * uses the current theme's colors and fonts
     */
    fun createTextField(columns: Int = 15): JTextField {
        return JTextField(columns).apply {
            font = fontBody
            background = inputBackground
            foreground = textPrimary
            caretColor = accentGreen
            border = BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(tableBorder, 1),
                EmptyBorder(8, 10, 8, 10)
            )
        }
    }

    /**
     * create a styled text area for multi-line user input
     * uses the current theme's colors and fonts
     * border matches text fields for consistency with tableBorder
     */
    fun createTextArea(rows: Int = 4, cols: Int = 20): JTextArea {
        return JTextArea(rows, cols).apply {
            font = fontBody
            background = inputBackground
            foreground = textPrimary
            caretColor = accentGreen
            lineWrap = true
            wrapStyleWord = true
            border = BorderFactory.createLineBorder(tableBorder, 1)
        }
    }

    /**
     * create a styled combo box for dropdown selections
     * uses the current theme's colors and fonts
     */
    fun createComboBox(): JComboBox<Any> {
        val combo = JComboBox<Any>()
        combo.font = fontBody
        combo.background = inputBackground
        combo.foreground = textPrimary
        return combo
    }

    /**
     * apply consistent styling to an existing combo box
     */
    fun styleComboBox(combo: JComboBox<*>) {
        combo.background = inputBackground
        combo.foreground = textPrimary
        combo.border = BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(tableBorder, 1),
            EmptyBorder(4, 8, 4, 8)
        )

        /**
         * Custom renderer to style dropdown items
         */
        combo.setRenderer(object : DefaultListCellRenderer() {
            override fun getListCellRendererComponent(
                list: JList<*>?,
                value: Any?,
                index: Int,
                isSelected: Boolean,
                cellHasFocus: Boolean
            ): Component {
                super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus)

                background = if (isSelected) tableSelection else inputBackground
                foreground = textPrimary

                border = EmptyBorder(5, 10, 5, 10)
                return this
            }
        })

        /**
         * ensure the editor component is also styled
         */
        val editor = combo.editor.editorComponent
        if (editor is JComponent) {
            editor.background = inputBackground
            editor.foreground = textPrimary
        }

        /**
         * repaint to apply changes
         */
        combo.repaint()
    }

    /**
     * create primary, secondary, danger, and accent button
     * buttons have rounded corners and hover effects
     * uses the current theme's colours and fonts
     */
    fun createPrimaryButton(text: String): JButton = createStyledButton(text, accentGreen, Color.BLACK)
    fun createSecondaryButton(text: String): JButton = createStyledButton(text, Color(100, 100, 100), Color.WHITE)
    fun createDangerButton(text: String): JButton = createStyledButton(text, accentRed, Color.WHITE)
    fun createAccentButton(text: String): JButton = createStyledButton(text, accentBlue, Color.WHITE)

    /**
     * helper method to create a styled button with custom colours
     * applies rounded corners and hover effects
     */
    private fun createStyledButton(text: String, bgColor: Color, fgColor: Color): JButton {
        return object : JButton(text) {
            init {
                font = fontBold
                foreground = fgColor
                background = bgColor
                isFocusPainted = false
                isBorderPainted = false
                isContentAreaFilled = false
                cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
                border = EmptyBorder(10, 20, 10, 20)
            }
            override fun paintComponent(g: Graphics) {
                val g2 = g as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = if (model.isRollover) bgColor.brighter() else bgColor
                g2.fill(RoundRectangle2D.Float(0f, 0f, width.toFloat(), height.toFloat(), 20f, 20f))
                super.paintComponent(g)
            }
        }
    }


    /**
     * create a card panel
     * rounded corners and background in cardBackground colour
     * padding inside the panel for content
     * used throughout the app for consistent card styling
     */
    fun createCardPanel(): JPanel {
        return object : JPanel(BorderLayout()) {
            override fun paintComponent(g: Graphics) {
                val g2 = g as Graphics2D
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g2.color = cardBackground
                g2.fillRoundRect(0, 0, width, height, 16, 16)
            }
        }.apply {
            isOpaque = false
            border = EmptyBorder(20, 20, 20, 20)
        }
    }

    /**
     * apply consistent styling to a JTable
     * row height, fonts, colours, selection behavior
     * panel will call this method to style tables consistently
     */
    fun styleTable(table: JTable) {
        table.apply {
            background = cardBackground
            foreground = textPrimary
            selectionBackground = tableSelection
            selectionForeground = if(current.isDark) accentGreen else Color.BLACK
            gridColor = tableBorder
            rowHeight = 40
            font = fontBody
            setShowGrid(false)
            setShowHorizontalLines(true)
            intercellSpacing = Dimension(0, 1)
            fillsViewportHeight = true

            tableHeader.apply {
                background = cardBackground
                foreground = textMuted
                font = fontSection
                border = BorderFactory.createMatteBorder(0, 0, 2, 0, tableBorder)
                (defaultRenderer as? DefaultTableCellRenderer)?.horizontalAlignment = SwingConstants.LEFT
            }
        }
    }

    /**
     * create a styled JScrollPane for tables and lists
     * uses current theme's colours for border and background
     */
    fun createScrollPane(view: Component): JScrollPane {
        return JScrollPane(view).apply {
            border = BorderFactory.createLineBorder(tableBorder)
            viewport.background = cardBackground
            background = cardBackground
        }
    }
}