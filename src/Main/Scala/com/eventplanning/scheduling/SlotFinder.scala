package com.eventplanning.scheduling

import com.eventplanning.domain.Venue
import com.eventplanning.domain.Event
import java.time.LocalDateTime
import java.time.Duration
import scala.annotation.tailrec
import scala.jdk.CollectionConverters._

object SlotFinder {

  /**
   * We set it so it looks for available slots within the next 7 days
   * in 15-minute increments
   */
  private val SEARCH_DAYS = 7
  private val SLOT_MINUTES = 15
  private val MAX_SLOT_SEARCH_ATTEMPTS: Int =
    SEARCH_DAYS * 24 * (60 / SLOT_MINUTES)

  /**
   * This is what ScalaBridge calls
   * validates inputs and finds available slots
   * converts Java lists to Scala lists and back
   * all venues that can accommodate the required capacity are considered
   * converts the result back to Java list of maps
   */
  private def findAvailableSlot(
                                 venues: java.util.List[Venue],
                                 existingEvents: java.util.List[Event],
                                 requiredCapacity: Int,
                                 earliestStartDate: LocalDateTime,
                                 duration: Duration
                               ): java.util.List[java.util.Map[String, Any]] = {

    /**
     * basic null and invalid input checks
     * if requiredCapacity is <= 0 or duration is negative or zero, return empty list
     * or if duration is zero or negative, return empty list
     */
    if (requiredCapacity <= 0 || duration.isNegative || duration.isZero) {
      return new java.util.ArrayList[java.util.Map[String, Any]]()
    }

    /**
     * convert Java lists to Scala lists
     */
    val venueList = venues.asScala.toList
    val eventList = existingEvents.asScala.toList

    /**
     * filter venues that can accommodate the required capacity
     */
    val capableVenues = venueList.filter(_.getCapacity >= requiredCapacity)

    val validSlots = findSlotsForVenues(capableVenues, eventList, earliestStartDate, duration, Nil)

    /**
     * sort slots by start time (earliest first)
     */
    val sortedSlots = validSlots.sortWith { (a, b) =>
      val startA = a.get("start").asInstanceOf[LocalDateTime]
      val startB = b.get("start").asInstanceOf[LocalDateTime]
      startA.isBefore(startB)
    }

    /**
     * convert back to Java list
     */
    sortedSlots.asJava
  }

  /**
   *This is the method called by ScalaBridge
   * It simply forwards the parameters to `findAvailableSlot`
   * returns the result unchanged.
   */
  def findAllAvailableSlots(
                             venues: java.util.List[Venue],
                             existingEvents: java.util.List[Event],
                             requiredCapacity: Int,
                             earliestStartDate: LocalDateTime,
                             duration: Duration
                           ): java.util.List[java.util.Map[String, Any]] = {
    findAvailableSlot(venues, existingEvents, requiredCapacity, earliestStartDate, duration)
  }

  /**
   *this is a tail-recursive method
   * it takes the list of remaining venues to check
   * for each venue, it tries to find the next free slot
   * if a slot is found, it calculates how long the venue is free from that start
   */
  @tailrec
  private def findSlotsForVenues(
                                  remainingVenues: List[Venue],
                                  allEvents: List[Event],
                                  requestedStart: LocalDateTime,
                                  duration: Duration,
                                  acc: List[java.util.Map[String, Any]]
                                ): List[java.util.Map[String, Any]] = {
    remainingVenues match {

      /**
       * base case: no more venues to check, return accumulated slots
       */
      case Nil => acc

      /**
       * if there are venues left to check
       * try to find the next free slot by searching foward from requestedStart
       * if found, calculate how long the venue is free from that start
       */
      case venue :: tail =>
        findNextFreeSlot(venue, allEvents, requestedStart, duration, 0) match {
          case Some(foundStart) =>
            val freeUntil = calculateFreeUntil(venue, allEvents, foundStart)

            /**
             * create a map with venue, start, and end (freeUntil)
             */
            val map = new java.util.HashMap[String, Any]()
            map.put("venue", venue)
            map.put("start", foundStart)
            map.put("end", freeUntil)

            /**
             * continue with the rest of the venues after adding this slot to the accumulator
             */
            findSlotsForVenues(tail, allEvents, requestedStart, duration, map :: acc)

          /**
           * if no slot found for this venue, continue with the rest
           */
          case None =>
            findSlotsForVenues(tail, allEvents, requestedStart, duration, acc)
        }
    }
  }

  /**
   *another tail-recursive method
   * start from currentStart and move forward in SLOT_MINUTES increments
   * finds if the venue is free for the given duration
   * stops after MAX_SLOT_SEARCH_ATTEMPTS to avoid infinite loops
   */
  @tailrec
  private def findNextFreeSlot(
                                venue: Venue,
                                events: List[Event],
                                currentStart: LocalDateTime,
                                duration: Duration,
                                attempt: Int
                              ): Option[LocalDateTime] = {

    if (attempt > MAX_SLOT_SEARCH_ATTEMPTS) {
      None

    } else if (isVenueFree(venue, events, currentStart, duration)) {
      Some(currentStart)

    } else {
      findNextFreeSlot(
        venue,
        events,
        currentStart.plusMinutes(SLOT_MINUTES.toLong),
        duration,
        attempt + 1
      )
    }
  }

  /**
   *Calculates until when the venue is free starting from startTime
   */
  private def calculateFreeUntil(venue: Venue, events: List[Event], startTime: LocalDateTime): LocalDateTime = {
    val nextEvent = events

      /**
      *filter event to only those in the same venue
      * filter events that are after the startTime
      */
      .filter(e => e.getVenue.getId == venue.getId)
      .filter(e => e.getDateTime.isAfter(startTime))
      .sortBy(_.getDateTime)
      .headOption

    nextEvent match {
      case Some(event) => event.getDateTime
      case None        => startTime.plusHours(24)
    }
  }

  /**
  *checks if the venue is free for the given duration starting from start
   * returns true if free, false if there is a conflict
   */
  private def isVenueFree(venue: Venue, events: List[Event], start: LocalDateTime, duration: Duration): Boolean = {
    val end = start.plus(duration)
    !events.exists { event =>
      event.getVenue.getId == venue.getId &&
        eventsOverlap(event.getDateTime, event.getEndTime, start, end)
    }
  }

  /**
  *check if two time intervals overlap
   * return true if: start1 < end2 && end1 > start2
   */
  private def eventsOverlap(start1: LocalDateTime, end1: LocalDateTime, start2: LocalDateTime, end2: LocalDateTime): Boolean = {
    start1.isBefore(end2) && end1.isAfter(start2)
  }
}